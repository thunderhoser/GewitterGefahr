"""Unit tests for link_winds_to_storms.py."""

import unittest
import numpy
import pandas
from gewittergefahr.gg_io import segmotion_io
from gewittergefahr.linkage import link_winds_to_storms as link_winds

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

# The following constants are used in `test_filter_storms_by_time*`.
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

# The following constants are used in `test_interp_one_storm_in_time*`.
STORM_ID_FOR_INTERP = 'ef12_hypercane'
STORM_TIMES_UNIX_SEC = numpy.array([0, 300, 600])
STORM_CENTROIDS_X_METRES = numpy.array([5000., 10000., 12000.])
STORM_CENTROIDS_Y_METRES = numpy.array([5000., 6000., 9000.])
SQUARE_X_METRES = numpy.array([0., 10000., 10000., 0., 0.])
SQUARE_Y_METRES = numpy.array([0., 0., 10000., 10000., 0.])
TALL_RECTANGLE_X_METRES = numpy.array([5000., 15000., 15000., 5000., 5000.])
TALL_RECTANGLE_Y_METRES = numpy.array([-4000., -4000., 16000., 16000., -4000.])
WIDE_RECTANGLE_X_METRES = numpy.array([2000., 22000., 22000., 2000., 2000.])
WIDE_RECTANGLE_Y_METRES = numpy.array([4000., 4000., 14000., 14000., 4000.])

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

STORM_OBJECT_TABLE_1CELL[link_winds.VERTICES_X_COLUMN].values[0] = (
    SQUARE_X_METRES)
STORM_OBJECT_TABLE_1CELL[link_winds.VERTICES_X_COLUMN].values[1] = (
    TALL_RECTANGLE_X_METRES)
STORM_OBJECT_TABLE_1CELL[link_winds.VERTICES_X_COLUMN].values[2] = (
    WIDE_RECTANGLE_X_METRES)
STORM_OBJECT_TABLE_1CELL[link_winds.VERTICES_Y_COLUMN].values[0] = (
    SQUARE_Y_METRES)
STORM_OBJECT_TABLE_1CELL[link_winds.VERTICES_Y_COLUMN].values[1] = (
    TALL_RECTANGLE_Y_METRES)
STORM_OBJECT_TABLE_1CELL[link_winds.VERTICES_Y_COLUMN].values[2] = (
    WIDE_RECTANGLE_Y_METRES)

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

# The following constants are used in test_interp_storms_in_time.
STORM_IDS_FOR_INTERP = ['ef12_hypercane', 'category6',
                        'ef12_hypercane', 'category6',
                        'ef12_hypercane', 'category6']
STORM_TIMES_UNIX_SEC = numpy.array([0, 0,
                                    300, 300,
                                    700, 600])
STORM_CENTROIDS_X_METRES = numpy.array([5000., 5000.,
                                        10000., 10000.,
                                        12000., 12000.])
STORM_CENTROIDS_Y_METRES = numpy.array([5000., 5000.,
                                        6000., 6000.,
                                        9000., 9000.])
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

STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_X_COLUMN].values[0] = (
    SQUARE_X_METRES)
STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_X_COLUMN].values[1] = (
    SQUARE_X_METRES)
STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_X_COLUMN].values[2] = (
    TALL_RECTANGLE_X_METRES)
STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_X_COLUMN].values[3] = (
    TALL_RECTANGLE_X_METRES)
STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_X_COLUMN].values[4] = (
    WIDE_RECTANGLE_X_METRES)
STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_X_COLUMN].values[5] = (
    WIDE_RECTANGLE_X_METRES)
STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_Y_COLUMN].values[0] = (
    SQUARE_Y_METRES)
STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_Y_COLUMN].values[1] = (
    SQUARE_Y_METRES)
STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_Y_COLUMN].values[2] = (
    TALL_RECTANGLE_Y_METRES)
STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_Y_COLUMN].values[3] = (
    TALL_RECTANGLE_Y_METRES)
STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_Y_COLUMN].values[4] = (
    WIDE_RECTANGLE_Y_METRES)
STORM_OBJECT_TABLE_2CELLS[link_winds.VERTICES_Y_COLUMN].values[5] = (
    WIDE_RECTANGLE_Y_METRES)

QUERY_TIME_UNIX_SEC = 600
MAX_TIME_BEFORE_START_SEC = 0
MAX_TIME_AFTER_END_SEC = 0

INTERP_VERTICES_X_METRES = numpy.array([2000., 22000., 22000., 2000., 2000.,
                                        1500., 21500., 21500., 1500., 1500.])
INTERP_VERTICES_Y_METRES = numpy.array([4000., 4000., 14000., 14000., 4000.,
                                        3250., 3250., 13250., 13250., 3250.])
STORM_ID_LIST = [
    'category6', 'category6', 'category6', 'category6', 'category6',
    'ef12_hypercane', 'ef12_hypercane', 'ef12_hypercane', 'ef12_hypercane',
    'ef12_hypercane']

VERTEX_DICT_2OBJECTS = {
    segmotion_io.STORM_ID_COLUMN: STORM_ID_LIST,
    link_winds.VERTEX_X_COLUMN: INTERP_VERTICES_X_METRES,
    link_winds.VERTEX_Y_COLUMN: INTERP_VERTICES_Y_METRES}
VERTEX_TABLE_2OBJECTS = pandas.DataFrame.from_dict(VERTEX_DICT_2OBJECTS)


class LinkWindsToStormsTests(unittest.TestCase):
    """Each method is a unit test for link_winds_to_storms.py."""

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
        self.assertTrue(this_vertex_table.equals(VERTEX_TABLE_2OBJECTS))


if __name__ == '__main__':
    unittest.main()
