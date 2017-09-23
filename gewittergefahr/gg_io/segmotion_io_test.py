"""Unit tests for segmotion_io.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_io import segmotion_io

TOLERANCE = 1e-6
XML_COLUMN_NAME_ORIG = segmotion_io.NORTH_VELOCITY_COLUMN_ORIG
XML_COLUMN_NAME = segmotion_io.NORTH_VELOCITY_COLUMN
STATS_COLUMN_NAMES = [segmotion_io.EAST_VELOCITY_COLUMN,
                      segmotion_io.NORTH_VELOCITY_COLUMN,
                      segmotion_io.START_TIME_COLUMN, segmotion_io.AGE_COLUMN]

MIN_BUFFER_DISTS_METRES = numpy.array([numpy.nan, 0., 5000.])
MAX_BUFFER_DISTS_METRES = numpy.array([0., 5000., 10000.])
BUFFER_LAT_COLUMN_NAMES = ['vertex_latitudes_deg_buffer_0m',
                           'vertex_latitudes_deg_buffer_0_5000m',
                           'vertex_latitudes_deg_buffer_5000_10000m']
BUFFER_LNG_COLUMN_NAMES = ['vertex_longitudes_deg_buffer_0m',
                           'vertex_longitudes_deg_buffer_0_5000m',
                           'vertex_longitudes_deg_buffer_5000_10000m']

TIME_STRING = '20170910-181300'
UNIX_TIME_SEC = 1505067180

STORM_IDS = ['0', '1', '2', '3', '4', None]
EAST_VELOCITIES_M_S01 = numpy.array([5., 7.5, 10., 8., 2.5, 4.])
NORTH_VELOCITIES_M_S01 = numpy.array([6., 9., numpy.nan, 3., -3., 4.])
START_TIMES_UNIX_SEC = numpy.array(
    [1505259103, 1505259103, 1505259103, 1505259103, 1505259103, 1505259103])
STORM_AGES_SEC = numpy.array([3000, numpy.nan, 2700, 2450, 1000, 3300])

STATS_DICT = {segmotion_io.STORM_ID_COLUMN: STORM_IDS,
              segmotion_io.EAST_VELOCITY_COLUMN: EAST_VELOCITIES_M_S01,
              segmotion_io.NORTH_VELOCITY_COLUMN: NORTH_VELOCITIES_M_S01,
              segmotion_io.START_TIME_COLUMN: START_TIMES_UNIX_SEC,
              segmotion_io.AGE_COLUMN: STORM_AGES_SEC}
STATS_TABLE_WITH_NANS = pandas.DataFrame.from_dict(STATS_DICT)

NAN_ROWS = [1, 2, 5]
STATS_TABLE_WITHOUT_NANS = STATS_TABLE_WITH_NANS.drop(
    STATS_TABLE_WITH_NANS.index[NAN_ROWS])

NUMERIC_STORM_ID_MATRIX = numpy.array(
    [[0, 0, 0, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan],
     [numpy.nan, 0, 0, 0, numpy.nan, numpy.nan, 1, 1],
     [numpy.nan, numpy.nan, 0, 0, numpy.nan, 1, 1, 1],
     [numpy.nan, numpy.nan, numpy.nan, numpy.nan, 1, 1, 1, 1],
     [numpy.nan, 2, 2, numpy.nan, 1, numpy.nan, numpy.nan, numpy.nan],
     [numpy.nan, 2, 2, 2, numpy.nan, numpy.nan, numpy.nan, 3]])

UNIQUE_CENTER_LAT_DEG = numpy.array([35., 35.1, 35.2, 35.3, 35.4, 35.5])
UNIQUE_CENTER_LNG_DEG = numpy.array(
    [262.7, 262.8, 262.9, 263., 263.1, 263.2, 263.3, 263.4])

UNIQUE_STORM_IDS = ['0', '1', '2', '3']
EXPECTED_POLYGON_DICT = {segmotion_io.STORM_ID_COLUMN: UNIQUE_STORM_IDS}
EXPECTED_POLYGON_TABLE = pandas.DataFrame.from_dict(EXPECTED_POLYGON_DICT)

STORM_ID_COLUMN = segmotion_io.STORM_ID_COLUMN
GRID_POINT_ROW_COLUMN = segmotion_io.GRID_POINT_ROW_COLUMN
GRID_POINT_COLUMN_COLUMN = segmotion_io.GRID_POINT_COLUMN_COLUMN
GRID_POINT_LAT_COLUMN = segmotion_io.GRID_POINT_LAT_COLUMN
GRID_POINT_LNG_COLUMN = segmotion_io.GRID_POINT_LNG_COLUMN

NESTED_ARRAY = EXPECTED_POLYGON_TABLE[
    [STORM_ID_COLUMN, STORM_ID_COLUMN]].values.tolist()
ARGUMENT_DICT = {GRID_POINT_LAT_COLUMN: NESTED_ARRAY,
                 GRID_POINT_LNG_COLUMN: NESTED_ARRAY,
                 GRID_POINT_ROW_COLUMN: NESTED_ARRAY,
                 GRID_POINT_COLUMN_COLUMN: NESTED_ARRAY}
EXPECTED_POLYGON_TABLE = EXPECTED_POLYGON_TABLE.assign(**ARGUMENT_DICT)

EXPECTED_POLYGON_TABLE[GRID_POINT_ROW_COLUMN].values[0] = numpy.array(
    [0, 0, 0, 1, 1, 1, 2, 2])
EXPECTED_POLYGON_TABLE[GRID_POINT_COLUMN_COLUMN].values[0] = numpy.array(
    [0, 1, 2, 1, 2, 3, 2, 3])
EXPECTED_POLYGON_TABLE[GRID_POINT_LAT_COLUMN].values[0] = numpy.array(
    [35., 35., 35., 35.1, 35.1, 35.1, 35.2, 35.2])
EXPECTED_POLYGON_TABLE[GRID_POINT_LNG_COLUMN].values[0] = numpy.array(
    [262.7, 262.8, 262.9, 262.8, 262.9, 263., 262.9, 263.])

EXPECTED_POLYGON_TABLE[GRID_POINT_ROW_COLUMN].values[1] = numpy.array(
    [1, 1, 2, 2, 2, 3, 3, 3, 3, 4])
EXPECTED_POLYGON_TABLE[GRID_POINT_COLUMN_COLUMN].values[1] = numpy.array(
    [6, 7, 5, 6, 7, 4, 5, 6, 7, 4])
EXPECTED_POLYGON_TABLE[GRID_POINT_LAT_COLUMN].values[1] = numpy.array(
    [35.1, 35.1, 35.2, 35.2, 35.2, 35.3, 35.3, 35.3, 35.3, 35.4])
EXPECTED_POLYGON_TABLE[GRID_POINT_LNG_COLUMN].values[1] = numpy.array(
    [263.3, 263.4, 263.2, 263.3, 263.4, 263.1, 263.2, 263.3, 263.4, 263.1])

EXPECTED_POLYGON_TABLE[GRID_POINT_ROW_COLUMN].values[2] = numpy.array(
    [4, 4, 5, 5, 5])
EXPECTED_POLYGON_TABLE[GRID_POINT_COLUMN_COLUMN].values[2] = numpy.array(
    [1, 2, 1, 2, 3])
EXPECTED_POLYGON_TABLE[GRID_POINT_LAT_COLUMN].values[2] = numpy.array(
    [35.4, 35.4, 35.5, 35.5, 35.5])
EXPECTED_POLYGON_TABLE[GRID_POINT_LNG_COLUMN].values[2] = numpy.array(
    [262.8, 262.9, 262.8, 262.9, 263.])

EXPECTED_POLYGON_TABLE[GRID_POINT_ROW_COLUMN].values[3] = numpy.array([5])
EXPECTED_POLYGON_TABLE[GRID_POINT_COLUMN_COLUMN].values[3] = numpy.array([7])
EXPECTED_POLYGON_TABLE[GRID_POINT_LAT_COLUMN].values[3] = numpy.array([35.5])
EXPECTED_POLYGON_TABLE[GRID_POINT_LNG_COLUMN].values[3] = numpy.array([263.4])

STATS_TABLE_INDICES_TO_JOIN = range(4)
STATS_TABLE_TO_JOIN = STATS_TABLE_WITH_NANS.loc[STATS_TABLE_INDICES_TO_JOIN]

EXPECTED_JOINED_TABLE = copy.deepcopy(EXPECTED_POLYGON_TABLE)
ARGUMENT_DICT = {segmotion_io.EAST_VELOCITY_COLUMN:
                     EAST_VELOCITIES_M_S01[STATS_TABLE_INDICES_TO_JOIN],
                 segmotion_io.NORTH_VELOCITY_COLUMN:
                     NORTH_VELOCITIES_M_S01[STATS_TABLE_INDICES_TO_JOIN],
                 segmotion_io.START_TIME_COLUMN:
                     START_TIMES_UNIX_SEC[STATS_TABLE_INDICES_TO_JOIN],
                 segmotion_io.AGE_COLUMN:
                     STORM_AGES_SEC[STATS_TABLE_INDICES_TO_JOIN]}
EXPECTED_JOINED_TABLE = EXPECTED_JOINED_TABLE.assign(**ARGUMENT_DICT)


class SegmotionIoTests(unittest.TestCase):
    """Each method is a unit test for segmotion_io.py."""

    def test_convert_xml_column_name(self):
        """Ensures correct output from _convert_xml_column_name."""

        this_xml_column_name = segmotion_io._convert_xml_column_name(
            XML_COLUMN_NAME_ORIG)
        self.assertTrue(this_xml_column_name == XML_COLUMN_NAME)

    def test_time_string_to_unix_sec(self):
        """Ensures correct output from _time_string_to_unix_sec."""

        this_time_unix_sec = segmotion_io._time_string_to_unix_sec(TIME_STRING)
        self.assertTrue(this_time_unix_sec == UNIX_TIME_SEC)

    def test_remove_rows_with_nan(self):
        """Ensures correct output from _remove_rows_with_nan."""

        this_stats_table = segmotion_io._remove_rows_with_nan(
            STATS_TABLE_WITH_NANS)
        self.assertTrue(this_stats_table.equals(STATS_TABLE_WITHOUT_NANS))

    def test_storm_id_matrix_to_coord_lists(self):
        """Ensures correct output from _storm_id_matrix_to_coord_lists."""

        mask_table = segmotion_io._storm_id_matrix_to_coord_lists(
            NUMERIC_STORM_ID_MATRIX, UNIQUE_CENTER_LAT_DEG,
            UNIQUE_CENTER_LNG_DEG)

        self.assertTrue(numpy.array_equal(mask_table[STORM_ID_COLUMN].values,
                                          EXPECTED_POLYGON_TABLE[
                                              STORM_ID_COLUMN].values))

        num_storms = len(mask_table.index)
        for i in range(num_storms):
            self.assertTrue(
                numpy.array_equal(mask_table[GRID_POINT_ROW_COLUMN].values[i],
                                  EXPECTED_POLYGON_TABLE[
                                      GRID_POINT_ROW_COLUMN].values[
                                      i]))
            self.assertTrue(
                numpy.array_equal(
                    mask_table[GRID_POINT_COLUMN_COLUMN].values[i],
                    EXPECTED_POLYGON_TABLE[
                        GRID_POINT_COLUMN_COLUMN].values[i]))
            self.assertTrue(
                numpy.allclose(mask_table[GRID_POINT_LAT_COLUMN].values[i],
                               EXPECTED_POLYGON_TABLE[
                                   GRID_POINT_LAT_COLUMN].values[i],
                               atol=TOLERANCE))
            self.assertTrue(
                numpy.allclose(mask_table[GRID_POINT_LNG_COLUMN].values[i],
                               EXPECTED_POLYGON_TABLE[
                                   GRID_POINT_LNG_COLUMN].values[i],
                               atol=TOLERANCE))

    def test_distance_buffers_to_column_names(self):
        """Ensures correct output from _distance_buffers_to_column_names."""

        (these_buffer_lat_column_names,
         these_buffer_lng_column_names) = (
            segmotion_io._distance_buffers_to_column_names(
                MIN_BUFFER_DISTS_METRES, MAX_BUFFER_DISTS_METRES))

        self.assertTrue(
            these_buffer_lat_column_names == BUFFER_LAT_COLUMN_NAMES)
        self.assertTrue(
            these_buffer_lng_column_names == BUFFER_LNG_COLUMN_NAMES)

    def test_join_stats_and_polygons(self):
        """Ensures correct output from join_stats_and_polygons."""

        joined_table = (
            segmotion_io.join_stats_and_polygons(EXPECTED_POLYGON_TABLE,
                                                 STATS_TABLE_TO_JOIN))

        joined_table_stats_only = joined_table[STATS_COLUMN_NAMES]
        exp_joined_table_stats_only = EXPECTED_JOINED_TABLE[STATS_COLUMN_NAMES]
        self.assertTrue(
            joined_table_stats_only.equals(exp_joined_table_stats_only))

        num_storms = len(joined_table.index)
        for i in range(num_storms):
            self.assertTrue(
                numpy.array_equal(joined_table[GRID_POINT_ROW_COLUMN].values[i],
                                  EXPECTED_JOINED_TABLE[
                                      GRID_POINT_ROW_COLUMN].values[
                                      i]))
            self.assertTrue(
                numpy.array_equal(
                    joined_table[GRID_POINT_COLUMN_COLUMN].values[i],
                    EXPECTED_JOINED_TABLE[
                        GRID_POINT_COLUMN_COLUMN].values[i]))
            self.assertTrue(
                numpy.allclose(joined_table[GRID_POINT_LAT_COLUMN].values[i],
                               EXPECTED_JOINED_TABLE[
                                   GRID_POINT_LAT_COLUMN].values[i],
                               atol=TOLERANCE))
            self.assertTrue(
                numpy.allclose(joined_table[GRID_POINT_LNG_COLUMN].values[i],
                               EXPECTED_JOINED_TABLE[
                                   GRID_POINT_LNG_COLUMN].values[i],
                               atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
