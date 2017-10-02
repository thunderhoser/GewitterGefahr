"""Unit tests for link_winds_to_storms.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_io import raw_wind_io
from gewittergefahr.gg_io import segmotion_io
from gewittergefahr.linkage import link_winds_to_storms as link_winds

# TODO(thunderhoser): change variable names.  Unit tests are hard to understand,
# because variable names make it unclear where (in which unit test) a variable
# being used.

TOLERANCE = 1e-6

# The following constants are used in test_storm_objects_to_cells.
STORM_IDS = ['a', 'b', 'c',
             'a', 'b', 'c', 'd',
             'a', 'c', 'd', 'e', 'f',
             'a', 'c', 'e', 'f',
             'a', 'e', 'f', 'g',
             'a', 'g']
VALID_TIMES_UNIX_SEC = numpy.array([0, 0, 0,
                                    300, 300, 300, 300,
                                    600, 600, 600, 600, 600,
                                    900, 900, 900, 900,
                                    1200, 1200, 1200, 1200,
                                    1500, 1500])
STORM_OBJECT_DICT = {segmotion_io.STORM_ID_COLUMN: STORM_IDS,
                     segmotion_io.TIME_COLUMN: VALID_TIMES_UNIX_SEC}
STORM_OBJECT_TABLE = pandas.DataFrame.from_dict(STORM_OBJECT_DICT)

START_TIMES_UNIX_SEC = numpy.array([0, 0, 0,
                                    0, 0, 0, 300,
                                    0, 0, 300, 600, 600,
                                    0, 0, 600, 600,
                                    0, 600, 600, 1200,
                                    0, 1200])
END_TIMES_UNIX_SEC = numpy.array([1500, 300, 900,
                                  1500, 300, 900, 600,
                                  1500, 900, 600, 1200, 1200,
                                  1500, 900, 1200, 1200,
                                  1500, 1200, 1200, 1500,
                                  1500, 1500])
ARGUMENT_DICT = {link_winds.START_TIME_COLUMN: START_TIMES_UNIX_SEC,
                 link_winds.END_TIME_COLUMN: END_TIMES_UNIX_SEC}
STORM_OBJECT_TABLE_WITH_CELL_INFO = STORM_OBJECT_TABLE.assign(**ARGUMENT_DICT)

# The following constants are used in test_filter_storms_by_time*.
EARLY_START_TIME_UNIX_SEC = 300
LATE_START_TIME_UNIX_SEC = 600
EARLY_END_TIME_UNIX_SEC = 900
LATE_END_TIME_UNIX_SEC = 1200

BAD_ROWS_EARLY_START_EARLY_END = numpy.array(
    [1, 4, 6, 9, 10, 11, 14, 15, 17, 18, 19, 21], dtype=int)
BAD_ROWS_EARLY_START_LATE_END = numpy.array(
    [1, 2, 4, 5, 6, 8, 9, 13, 10, 11, 14, 15, 17, 18, 19, 21], dtype=int)
BAD_ROWS_LATE_START_EARLY_END = numpy.array([1, 4, 6, 9, 19, 21], dtype=int)
BAD_ROWS_LATE_START_LATE_END = numpy.array(
    [1, 2, 4, 5, 6, 8, 9, 13, 19, 21], dtype=int)

STORM_OBJECT_TABLE_EARLY_START_EARLY_END = (
    STORM_OBJECT_TABLE_WITH_CELL_INFO.drop(
        STORM_OBJECT_TABLE_WITH_CELL_INFO.index[BAD_ROWS_EARLY_START_EARLY_END],
        axis=0, inplace=False))
STORM_OBJECT_TABLE_EARLY_START_LATE_END = (
    STORM_OBJECT_TABLE_WITH_CELL_INFO.drop(
        STORM_OBJECT_TABLE_WITH_CELL_INFO.index[BAD_ROWS_EARLY_START_LATE_END],
        axis=0, inplace=False))
STORM_OBJECT_TABLE_LATE_START_EARLY_END = (
    STORM_OBJECT_TABLE_WITH_CELL_INFO.drop(
        STORM_OBJECT_TABLE_WITH_CELL_INFO.index[BAD_ROWS_LATE_START_EARLY_END],
        axis=0, inplace=False))
STORM_OBJECT_TABLE_LATE_START_LATE_END = (
    STORM_OBJECT_TABLE_WITH_CELL_INFO.drop(
        STORM_OBJECT_TABLE_WITH_CELL_INFO.index[BAD_ROWS_LATE_START_LATE_END],
        axis=0, inplace=False))

# The following constants are used in test_interp_one_storm_in_time*.
STORM_ID_FOR_INTERP = 'ef12_hypercane'
STORM_TIMES_UNIX_SEC = numpy.array([0, 300, 600])
STORM_CENTROIDS_X_METRES = numpy.array([5000., 10000., 12000.])
STORM_CENTROIDS_Y_METRES = numpy.array([5000., 6000., 9000.])

STORM_OBJECT_DICT_1CELL = {
    segmotion_io.TIME_COLUMN: STORM_TIMES_UNIX_SEC,
    link_winds.CENTROID_X_COLUMN: STORM_CENTROIDS_X_METRES,
    link_winds.CENTROID_Y_COLUMN: STORM_CENTROIDS_Y_METRES}
STORM_OBJECT_TABLE_1CELL = pandas.DataFrame.from_dict(STORM_OBJECT_DICT_1CELL)

NESTED_ARRAY = STORM_OBJECT_TABLE_1CELL[[
    segmotion_io.TIME_COLUMN, segmotion_io.TIME_COLUMN]].values.tolist()
ARGUMENT_DICT = {link_winds.VERTICES_X_COLUMN: NESTED_ARRAY,
                 link_winds.VERTICES_Y_COLUMN: NESTED_ARRAY}
STORM_OBJECT_TABLE_1CELL = STORM_OBJECT_TABLE_1CELL.assign(**ARGUMENT_DICT)

STORM_OBJECT_TABLE_1CELL[link_winds.VERTICES_X_COLUMN].values[0] = numpy.array(
    [0., 10000., 10000., 0., 0.])
STORM_OBJECT_TABLE_1CELL[link_winds.VERTICES_X_COLUMN].values[1] = numpy.array(
    [5000., 15000., 15000., 5000., 5000.])
STORM_OBJECT_TABLE_1CELL[link_winds.VERTICES_X_COLUMN].values[2] = numpy.array(
    [2000., 22000., 22000., 2000., 2000.])
STORM_OBJECT_TABLE_1CELL[link_winds.VERTICES_Y_COLUMN].values[0] = numpy.array(
    [0., 0., 10000., 10000., 0.])
STORM_OBJECT_TABLE_1CELL[link_winds.VERTICES_Y_COLUMN].values[1] = numpy.array(
    [-4000., -4000., 16000., 16000., -4000.])
STORM_OBJECT_TABLE_1CELL[link_winds.VERTICES_Y_COLUMN].values[2] = numpy.array(
    [4000., 4000., 14000., 14000., 4000.])

TRUE_INTERP_TIME_UNIX_SEC = 375
EXTRAP_TIME_UNIX_SEC = 750

INTERP_VERTICES_X_METRES = numpy.array([5500., 15500., 15500., 5500., 5500.])
INTERP_VERTICES_Y_METRES = numpy.array([-3250., -3250., 16750., 16750., -3250.])
STORM_ID_LIST = ['ef12_hypercane', 'ef12_hypercane', 'ef12_hypercane',
                 'ef12_hypercane', 'ef12_hypercane']
VERTEX_DICT_1OBJECT_TRUE_INTERP = {
    segmotion_io.STORM_ID_COLUMN: STORM_ID_LIST,
    link_winds.VERTEX_X_COLUMN: INTERP_VERTICES_X_METRES,
    link_winds.VERTEX_Y_COLUMN: INTERP_VERTICES_Y_METRES}
VERTEX_TABLE_1OBJECT_TRUE_INTERP = pandas.DataFrame.from_dict(
    VERTEX_DICT_1OBJECT_TRUE_INTERP)

EXTRAP_VERTICES_X_METRES = numpy.array([3000., 23000., 23000., 3000., 3000.])
EXTRAP_VERTICES_Y_METRES = numpy.array([5500., 5500., 15500., 15500., 5500.])
VERTEX_DICT_1OBJECT_EXTRAP = {
    segmotion_io.STORM_ID_COLUMN: STORM_ID_LIST,
    link_winds.VERTEX_X_COLUMN: EXTRAP_VERTICES_X_METRES,
    link_winds.VERTEX_Y_COLUMN: EXTRAP_VERTICES_Y_METRES}
VERTEX_TABLE_1OBJECT_EXTRAP = pandas.DataFrame.from_dict(
    VERTEX_DICT_1OBJECT_EXTRAP)

# The following constants are used in test_get_xy_bounding_box_of_storms.
PADDING_FOR_BOUNDING_BOX_METRES = 1000.
BOUNDING_BOX_X_METRES = numpy.array([-1000., 23000.])
BOUNDING_BOX_Y_METRES = numpy.array([-5000., 17000.])

# The following constants are used in test_interp_storms_in_time.
STORM_IDS_FOR_INTERP = ['ef12_hypercane', 'category6',
                        'ef12_hypercane', 'category6',
                        'ef12_hypercane', 'category6']
STORM_TIMES_UNIX_SEC = numpy.array([0, 0,
                                    300, 300,
                                    700, 600])
STORM_CENTROIDS_X_METRES = numpy.array([-50000., 50000.,
                                        -48000, 55000.,
                                        -46000., 59000.])
STORM_CENTROIDS_Y_METRES = numpy.array([-50000., 50000.,
                                        -49000, 52000.,
                                        -48500., 55000.])
START_TIMES_UNIX_SEC = numpy.array([0, 0, 0, 0, 0, 0])
END_TIMES_UNIX_SEC = numpy.array([700, 600, 700, 600, 700, 600])

STORM_OBJECT_DICT_2CELLS = {
    segmotion_io.TIME_COLUMN: STORM_TIMES_UNIX_SEC,
    segmotion_io.STORM_ID_COLUMN: STORM_IDS_FOR_INTERP,
    link_winds.CENTROID_X_COLUMN: STORM_CENTROIDS_X_METRES,
    link_winds.CENTROID_Y_COLUMN: STORM_CENTROIDS_Y_METRES,
    link_winds.START_TIME_COLUMN: START_TIMES_UNIX_SEC,
    link_winds.END_TIME_COLUMN: END_TIMES_UNIX_SEC}
STORM_OBJECT_TABLE_2CELLS = pandas.DataFrame.from_dict(STORM_OBJECT_DICT_2CELLS)

NESTED_ARRAY = STORM_OBJECT_TABLE_2CELLS[[
    segmotion_io.TIME_COLUMN, segmotion_io.TIME_COLUMN]].values.tolist()
ARGUMENT_DICT = {link_winds.VERTICES_X_COLUMN: NESTED_ARRAY,
                 link_winds.VERTICES_Y_COLUMN: NESTED_ARRAY}
STORM_OBJECT_TABLE_2CELLS = STORM_OBJECT_TABLE_2CELLS.assign(**ARGUMENT_DICT)

STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_X_COLUMN].values[0] = numpy.array(
    [-55000., -45000., -45000., -55000., -55000.])
STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_X_COLUMN].values[1] = numpy.array(
    [45000., 55000., 55000., 45000., 45000.])
STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_X_COLUMN].values[2] = numpy.array(
    [-53000., -43000., -43000., -53000., -53000.])
STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_X_COLUMN].values[3] = numpy.array(
    [50000., 60000., 60000., 50000., 50000.])
STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_X_COLUMN].values[4] = numpy.array(
    [-56000., -36000., -36000., -56000., -56000.])
STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_X_COLUMN].values[5] = numpy.array(
    [49000., 69000., 69000., 49000., 49000.])
STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_Y_COLUMN].values[0] = numpy.array(
    [-55000., -55000., -45000., -45000., -55000.])
STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_Y_COLUMN].values[1] = numpy.array(
    [45000., 45000., 55000., 55000., 45000.])
STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_Y_COLUMN].values[2] = numpy.array(
    [-59000., -59000., -39000., -39000., -59000.])
STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_Y_COLUMN].values[3] = numpy.array(
    [42000., 42000., 62000., 62000., 42000.])
STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_Y_COLUMN].values[4] = numpy.array(
    [-53500., -53500., -43500., -43500., -53500.])
STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_Y_COLUMN].values[5] = numpy.array(
    [50000., 50000., 60000., 60000., 50000.])

QUERY_TIME_UNIX_SEC = 600
MAX_TIME_BEFORE_START_SEC = 0
MAX_TIME_AFTER_END_SEC = 0

INTERP_VERTICES_X_METRES = numpy.array(
    [49000., 69000., 69000., 49000., 49000.,
     -56500., -36500., -36500., -56500., -56500.])
INTERP_VERTICES_Y_METRES = numpy.array(
    [50000., 50000., 60000., 60000., 50000.,
     -53625., -53625., -43625., -43625., -53625.])
STORM_ID_LIST = [
    'category6', 'category6', 'category6', 'category6', 'category6',
    'ef12_hypercane', 'ef12_hypercane', 'ef12_hypercane', 'ef12_hypercane',
    'ef12_hypercane']

INTERP_VERTEX_DICT_2OBJECTS = {
    segmotion_io.STORM_ID_COLUMN: STORM_ID_LIST,
    link_winds.VERTEX_X_COLUMN: INTERP_VERTICES_X_METRES,
    link_winds.VERTEX_Y_COLUMN: INTERP_VERTICES_Y_METRES}
INTERP_VERTEX_TABLE_2OBJECTS = pandas.DataFrame.from_dict(
    INTERP_VERTEX_DICT_2OBJECTS)

# The following constants are used in test_filter_winds_by_bounding_box.
WIND_X_COORDS_METRES = numpy.array(
    [59000., 59000., 59000., 59000., 59000.,
     -46500., -36500., -31500., 0., 1e6])
WIND_Y_COORDS_METRES = numpy.array(
    [55000., 50000., 45000., 0., -1e6,
     -49000., -49000., -49000., -49000., -49000.])
WIND_X_LIMITS_METRES = numpy.array([-59000., 72000.])
WIND_Y_LIMITS_METRES = numpy.array([-62000., 65000.])

WIND_DICT = {link_winds.WIND_X_COLUMN: WIND_X_COORDS_METRES,
             link_winds.WIND_Y_COLUMN: WIND_Y_COORDS_METRES}
WIND_TABLE = pandas.DataFrame.from_dict(WIND_DICT)

WIND_ROWS_OUTSIDE_BOX = numpy.array([4, 9], dtype=int)
FILTERED_WIND_TABLE = WIND_TABLE.drop(
    WIND_TABLE.index[WIND_ROWS_OUTSIDE_BOX], axis=0, inplace=False)

# The following constants are used in test_find_nearest_storms_one_time.
WIND_X_COORDS_METRES = numpy.array([49000., 49000., 49000., 49000.,
                                    -46500., -36500., -31500., 0.])
WIND_Y_COORDS_METRES = numpy.array([55000., 50000., 45000., 0.,
                                    -43625., -43625., -43625., -43625.])

MAX_LINK_DISTANCE_METRES = 10000.
NEAREST_STORM_IDS_ONE_TIME = [
    'category6', 'category6', 'category6', None,
    'ef12_hypercane', 'ef12_hypercane', 'ef12_hypercane', None]
LINK_DISTANCES_METRES_ONE_TIME = numpy.array([0., 0., 5000., numpy.nan,
                                              0., 0., 5000., numpy.nan])

# The following constants are used in test_find_nearest_storms.
WIND_X_COORDS_METRES_MULTI_TIMES = numpy.array(
    [49000., 49000., 49000., 49000.,
     -46500., -36500., -31500., 0.,
     49000., 49000., 49000., 49000.,
     -46000., -36000., -31000., 0., ])
WIND_Y_COORDS_METRES_MULTI_TIMES = numpy.array(
    [55000., 50000., 45000., 0.,
     -43625., -43625., -43625., -43625.,
     55000., 50000., 45000., 0.,
     -43500., -43500., -43500., -43500.])
WIND_TIMES_UNIX_SEC = numpy.array([600, 600, 600, 600,
                                   600, 600, 600, 600,
                                   700, 700, 700, 700,
                                   700, 700, 700, 700])

WIND_DICT_MULTI_TIMES = {
    link_winds.WIND_X_COLUMN: WIND_X_COORDS_METRES_MULTI_TIMES,
    link_winds.WIND_Y_COLUMN: WIND_Y_COORDS_METRES_MULTI_TIMES,
    raw_wind_io.TIME_COLUMN: WIND_TIMES_UNIX_SEC}
WIND_TABLE_MULTI_TIMES = pandas.DataFrame.from_dict(WIND_DICT_MULTI_TIMES)

NEAREST_STORM_IDS_MULTI_TIMES = [
    'category6', 'category6', 'category6', None,
    'ef12_hypercane', 'ef12_hypercane', 'ef12_hypercane', None,
    None, None, None, None,
    'ef12_hypercane', 'ef12_hypercane', 'ef12_hypercane', None]
LINK_DISTANCES_METRES_MULTI_TIMES = numpy.array(
    [0., 0., 5000., numpy.nan,
     0., 0., 5000., numpy.nan,
     numpy.nan, numpy.nan, numpy.nan, numpy.nan,
     0., 0., 5000., numpy.nan])

ARGUMENT_DICT = {
    link_winds.NEAREST_STORM_ID_COLUMN: NEAREST_STORM_IDS_MULTI_TIMES,
    link_winds.LINKAGE_DISTANCE_COLUMN: LINK_DISTANCES_METRES_MULTI_TIMES}
WIND_TO_STORM_TABLE_MULTI_TIMES = WIND_TABLE_MULTI_TIMES.assign(**ARGUMENT_DICT)

# The following constants are used in test_link_winds_and_storms.
WIND_STATION_IDS = ['a', 'b', 'c', 'd',
                    'e', 'f', 'g', 'h',
                    'a', 'b', 'c', 'd',
                    'e', 'f', 'g', 'h']
WIND_LATITUDES_DEG = numpy.array([1., 2., 3., 4.,
                                  5., 6., 7., 8.,
                                  1., 2., 3., 4.,
                                  5., 6., 7., 8.])
WIND_LONGITUDES_DEG = copy.deepcopy(WIND_LATITUDES_DEG)
U_WINDS_M_S01 = copy.deepcopy(WIND_LATITUDES_DEG)
V_WINDS_M_S01 = copy.deepcopy(WIND_LATITUDES_DEG)

ARGUMENT_DICT = {raw_wind_io.STATION_ID_COLUMN: WIND_STATION_IDS,
                 raw_wind_io.LATITUDE_COLUMN: WIND_LATITUDES_DEG,
                 raw_wind_io.LONGITUDE_COLUMN: WIND_LONGITUDES_DEG,
                 raw_wind_io.U_WIND_COLUMN: U_WINDS_M_S01,
                 raw_wind_io.V_WIND_COLUMN: V_WINDS_M_S01}
FULL_WIND_TABLE_MULTI_TIMES = WIND_TO_STORM_TABLE_MULTI_TIMES.assign(
    **ARGUMENT_DICT)

LINKAGE_TABLE = copy.deepcopy(STORM_OBJECT_TABLE_2CELLS)
NESTED_ARRAY = LINKAGE_TABLE[[
    segmotion_io.TIME_COLUMN, segmotion_io.TIME_COLUMN]].values.tolist()
ARGUMENT_DICT = {link_winds.STATION_IDS_COLUMN: NESTED_ARRAY,
                 link_winds.WIND_LATS_COLUMN: NESTED_ARRAY,
                 link_winds.WIND_LNGS_COLUMN: NESTED_ARRAY,
                 link_winds.U_WINDS_COLUMN: NESTED_ARRAY,
                 link_winds.V_WINDS_COLUMN: NESTED_ARRAY,
                 link_winds.LINKAGE_DISTANCES_COLUMN: NESTED_ARRAY,
                 link_winds.RELATIVE_TIMES_COLUMN: NESTED_ARRAY}
LINKAGE_TABLE = LINKAGE_TABLE.assign(**ARGUMENT_DICT)

HYPERCANE_STATION_IDS = ['e', 'f', 'g', 'e', 'f', 'g']
HYPERCANE_WIND_LATITUDES_DEG = numpy.array([5., 6., 7., 5., 6., 7.])
HYPERCANE_WIND_LONGITUDES_DEG = numpy.array([5., 6., 7., 5., 6., 7.])
HYPERCANE_U_WINDS_M_S01 = numpy.array([5., 6., 7., 5., 6., 7.])
HYPERCANE_V_WINDS_M_S01 = numpy.array([5., 6., 7., 5., 6., 7.])
HYPERCANE_LINK_DISTANCES_METRES = numpy.array([0., 0., 5000., 0., 0., 5000.])

CATEGORY6_STATION_IDS = ['a', 'b', 'c']
CATEGORY6_WIND_LATITUDES_DEG = numpy.array([1., 2., 3.])
CATEGORY6_WIND_LONGITUDES_DEG = numpy.array([1., 2., 3.])
CATEGORY6_U_WINDS_M_S01 = numpy.array([1., 2., 3.])
CATEGORY6_V_WINDS_M_S01 = numpy.array([1., 2., 3.])
CATEGORY6_LINK_DISTANCES_METRES = numpy.array([0., 0., 5000.])

HYPERCANE_ROWS = numpy.array([0, 2, 4], dtype=int)
for this_row in HYPERCANE_ROWS:
    LINKAGE_TABLE[link_winds.STATION_IDS_COLUMN].values[this_row] = (
        HYPERCANE_STATION_IDS)
    LINKAGE_TABLE[link_winds.WIND_LATS_COLUMN].values[this_row] = (
        HYPERCANE_WIND_LATITUDES_DEG)
    LINKAGE_TABLE[link_winds.WIND_LNGS_COLUMN].values[this_row] = (
        HYPERCANE_WIND_LONGITUDES_DEG)
    LINKAGE_TABLE[link_winds.U_WINDS_COLUMN].values[this_row] = (
        HYPERCANE_U_WINDS_M_S01)
    LINKAGE_TABLE[link_winds.V_WINDS_COLUMN].values[this_row] = (
        HYPERCANE_V_WINDS_M_S01)
    LINKAGE_TABLE[link_winds.LINKAGE_DISTANCES_COLUMN].values[this_row] = (
        HYPERCANE_LINK_DISTANCES_METRES)

LINKAGE_TABLE[link_winds.RELATIVE_TIMES_COLUMN].values[0] = numpy.array(
    [600, 600, 600, 700, 700, 700])
LINKAGE_TABLE[link_winds.RELATIVE_TIMES_COLUMN].values[2] = numpy.array(
    [300, 300, 300, 400, 400, 400])
LINKAGE_TABLE[link_winds.RELATIVE_TIMES_COLUMN].values[4] = numpy.array(
    [-100, -100, -100, 0, 0, 0])

CATEGORY6_ROWS = numpy.array([1, 3, 5], dtype=int)
for this_row in CATEGORY6_ROWS:
    LINKAGE_TABLE[link_winds.STATION_IDS_COLUMN].values[this_row] = (
        CATEGORY6_STATION_IDS)
    LINKAGE_TABLE[link_winds.WIND_LATS_COLUMN].values[this_row] = (
        CATEGORY6_WIND_LATITUDES_DEG)
    LINKAGE_TABLE[link_winds.WIND_LNGS_COLUMN].values[this_row] = (
        CATEGORY6_WIND_LONGITUDES_DEG)
    LINKAGE_TABLE[link_winds.U_WINDS_COLUMN].values[this_row] = (
        CATEGORY6_U_WINDS_M_S01)
    LINKAGE_TABLE[link_winds.V_WINDS_COLUMN].values[this_row] = (
        CATEGORY6_V_WINDS_M_S01)
    LINKAGE_TABLE[link_winds.LINKAGE_DISTANCES_COLUMN].values[this_row] = (
        CATEGORY6_LINK_DISTANCES_METRES)

LINKAGE_TABLE[link_winds.RELATIVE_TIMES_COLUMN].values[1] = numpy.array(
    [600, 600, 600])
LINKAGE_TABLE[link_winds.RELATIVE_TIMES_COLUMN].values[3] = numpy.array(
    [300, 300, 300])
LINKAGE_TABLE[link_winds.RELATIVE_TIMES_COLUMN].values[5] = numpy.array(
    [0, 0, 0])


class LinkWindsToStormsTests(unittest.TestCase):
    """Each method is a unit test for link_winds_to_storms.py."""

    def test_get_xy_bounding_box_of_storms(self):
        """Ensures correct output from _get_xy_bounding_box_of_storms."""

        (these_x_limits_metres,
         these_y_limits_metres) = link_winds._get_xy_bounding_box_of_storms(
             STORM_OBJECT_TABLE_1CELL,
             padding_metres=PADDING_FOR_BOUNDING_BOX_METRES)

        self.assertTrue(numpy.allclose(
            these_x_limits_metres, BOUNDING_BOX_X_METRES, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_y_limits_metres, BOUNDING_BOX_Y_METRES, atol=TOLERANCE))

    def test_filter_winds_by_bounding_box(self):
        """Ensures correct output from _filter_winds_by_bounding_box."""

        this_filtered_wind_table = link_winds._filter_winds_by_bounding_box(
            WIND_TABLE, x_limits_metres=WIND_X_LIMITS_METRES,
            y_limits_metres=WIND_Y_LIMITS_METRES)
        self.assertTrue(this_filtered_wind_table.equals(FILTERED_WIND_TABLE))

    def test_storm_objects_to_cells(self):
        """Ensures correct output from _storm_objects_to_cells."""

        this_storm_object_table = link_winds._storm_objects_to_cells(
            STORM_OBJECT_TABLE)
        self.assertTrue(this_storm_object_table.equals(
            STORM_OBJECT_TABLE_WITH_CELL_INFO))

    def test_filter_storms_by_time_early_start_early_end(self):
        """Ensures correct output from _filter_storms_by_time.

        In this case, storm cells will be dropped if they start after 300
        seconds or end before 900 seconds.
        """

        this_storm_object_table = link_winds._filter_storms_by_time(
            STORM_OBJECT_TABLE_WITH_CELL_INFO,
            max_start_time_unix_sec=EARLY_START_TIME_UNIX_SEC,
            min_end_time_unix_sec=EARLY_END_TIME_UNIX_SEC)
        self.assertTrue(this_storm_object_table.equals(
            STORM_OBJECT_TABLE_EARLY_START_EARLY_END))

    def test_filter_storms_by_time_early_start_late_end(self):
        """Ensures correct output from _filter_storms_by_time.

        In this case, storm cells will be dropped if they start after 300
        seconds or end before 1200 seconds.
        """

        this_storm_object_table = link_winds._filter_storms_by_time(
            STORM_OBJECT_TABLE_WITH_CELL_INFO,
            max_start_time_unix_sec=EARLY_START_TIME_UNIX_SEC,
            min_end_time_unix_sec=LATE_END_TIME_UNIX_SEC)
        self.assertTrue(this_storm_object_table.equals(
            STORM_OBJECT_TABLE_EARLY_START_LATE_END))

    def test_filter_storms_by_time_late_start_early_end(self):
        """Ensures correct output from _filter_storms_by_time.

        In this case, storm cells will be dropped if they start after 600
        seconds or end before 900 seconds.
        """

        this_storm_object_table = link_winds._filter_storms_by_time(
            STORM_OBJECT_TABLE_WITH_CELL_INFO,
            max_start_time_unix_sec=LATE_START_TIME_UNIX_SEC,
            min_end_time_unix_sec=EARLY_END_TIME_UNIX_SEC)
        self.assertTrue(this_storm_object_table.equals(
            STORM_OBJECT_TABLE_LATE_START_EARLY_END))

    def test_filter_storms_by_time_late_start_late_end(self):
        """Ensures correct output from _filter_storms_by_time.

        In this case, storm cells will be dropped if they start after 600
        seconds or end before 1200 seconds.
        """

        this_storm_object_table = link_winds._filter_storms_by_time(
            STORM_OBJECT_TABLE_WITH_CELL_INFO,
            max_start_time_unix_sec=LATE_START_TIME_UNIX_SEC,
            min_end_time_unix_sec=LATE_END_TIME_UNIX_SEC)
        self.assertTrue(this_storm_object_table.equals(
            STORM_OBJECT_TABLE_LATE_START_LATE_END))

    def test_interp_one_storm_in_time_true_interp(self):
        """Ensures correct output from _interp_one_storm_in_time.

        In this case, doing true interpolation (no extrapolation).
        """

        this_vertex_table = link_winds._interp_one_storm_in_time(
            STORM_OBJECT_TABLE_1CELL, storm_id=STORM_ID_FOR_INTERP,
            query_time_unix_sec=TRUE_INTERP_TIME_UNIX_SEC)
        self.assertTrue(this_vertex_table.equals(
            VERTEX_TABLE_1OBJECT_TRUE_INTERP))

    def test_interp_one_storm_in_time_extrap(self):
        """Ensures correct output from _interp_one_storm_in_time.

        In this case, doing extrapolation.
        """

        this_vertex_table = link_winds._interp_one_storm_in_time(
            STORM_OBJECT_TABLE_1CELL, storm_id=STORM_ID_FOR_INTERP,
            query_time_unix_sec=EXTRAP_TIME_UNIX_SEC)
        self.assertTrue(this_vertex_table.equals(VERTEX_TABLE_1OBJECT_EXTRAP))

    def test_interp_storms_in_time(self):
        """Ensures correct output from _interp_storms_in_time."""

        this_vertex_table = link_winds._interp_storms_in_time(
            STORM_OBJECT_TABLE_2CELLS, query_time_unix_sec=QUERY_TIME_UNIX_SEC,
            max_time_before_start_sec=MAX_TIME_BEFORE_START_SEC,
            max_time_after_end_sec=MAX_TIME_AFTER_END_SEC)
        self.assertTrue(this_vertex_table.equals(INTERP_VERTEX_TABLE_2OBJECTS))

    def test_find_nearest_storms_one_time(self):
        """Ensures correct output from _find_nearest_storms_one_time."""

        (these_nearest_storm_ids,
         these_link_dist_metres) = link_winds._find_nearest_storms_one_time(
             INTERP_VERTEX_TABLE_2OBJECTS,
             wind_x_coords_metres=WIND_X_COORDS_METRES,
             wind_y_coords_metres=WIND_Y_COORDS_METRES,
             max_distance_metres=MAX_LINK_DISTANCE_METRES)

        self.assertTrue(these_nearest_storm_ids == NEAREST_STORM_IDS_ONE_TIME)
        self.assertTrue(numpy.allclose(
            these_link_dist_metres, LINK_DISTANCES_METRES_ONE_TIME,
            equal_nan=True, atol=TOLERANCE))

    def test_find_nearest_storms(self):
        """Ensures correct output from _find_nearest_storms."""

        this_wind_to_storm_table = link_winds._find_nearest_storms(
            storm_object_table=STORM_OBJECT_TABLE_2CELLS,
            wind_table=WIND_TABLE_MULTI_TIMES,
            max_time_before_storm_start_sec=MAX_TIME_BEFORE_START_SEC,
            max_time_after_storm_end_sec=MAX_TIME_AFTER_END_SEC,
            max_link_dist_metres=MAX_LINK_DISTANCE_METRES)

        self.assertTrue(this_wind_to_storm_table.equals(
            WIND_TO_STORM_TABLE_MULTI_TIMES))

    def test_link_winds_and_storms(self):
        """Ensures correct output from _link_winds_and_storms."""

        this_linkage_table = link_winds._link_winds_and_storms(
            storm_object_table=STORM_OBJECT_TABLE_2CELLS,
            wind_table=FULL_WIND_TABLE_MULTI_TIMES)

        these_column_names = list(this_linkage_table)
        expected_column_names = list(LINKAGE_TABLE)
        self.assertTrue(these_column_names == expected_column_names)

        num_rows = len(this_linkage_table.index)
        string_columns = [segmotion_io.STORM_ID_COLUMN,
                          link_winds.STATION_IDS_COLUMN]

        for i in range(num_rows):
            for this_column_name in these_column_names:
                if this_column_name in string_columns:
                    self.assertTrue(
                        this_linkage_table[this_column_name].values[i] ==
                        LINKAGE_TABLE[this_column_name].values[i])
                else:
                    self.assertTrue(numpy.allclose(
                        this_linkage_table[this_column_name].values[i],
                        LINKAGE_TABLE[this_column_name].values[i],
                        atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
