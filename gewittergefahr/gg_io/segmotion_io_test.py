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

UNIX_TIME_SEC = 1505067180
TIME_STRING = '20170910-181300'
PATHLESS_STATS_FILE_NAME_ZIPPED = '20170910-181300.xml.gz'
PATHLESS_POLYGON_FILE_NAME_ZIPPED = '20170910-181300.netcdf.gz'
PATHLESS_STATS_FILE_NAME_UNZIPPED = '20170910-181300.xml'
PATHLESS_POLYGON_FILE_NAME_UNZIPPED = '20170910-181300.netcdf'
PATHLESS_PROCESSED_FILE_NAME = 'segmotion_2017-09-10-181300.p'

SPC_DATE_STRING = '20170910'
SPC_DATE_UNIX_SEC = 1505066400
STORM_IDS_NO_SPC_DATE = ['989', '657', '212']
STORM_IDS_WITH_SPC_DATE = ['989_20170910', '657_20170910', '212_20170910']

TRACKING_SCALE_ORDINAL = 0
TRACKING_SCALE_METRES2 = 5e7
RELATIVE_STATS_DIR_NAME_ORDINAL_SCALE = '20170910/TrackingTable/scale_0'
RELATIVE_POLYGON_DIR_NAME_ORDINAL_SCALE = '20170910/ClusterID/scale_0'
RELATIVE_STATS_DIR_NAME_PHYSICAL_SCALE = (
    '20170910/TrackingTable/scale_50000000m2')
RELATIVE_POLYGON_DIR_NAME_PHYSICAL_SCALE = '20170910/ClusterID/scale_50000000m2'
RELATIVE_PROCESSED_DIR_NAME = '20170910/scale_50000000m2'

TOP_RAW_DIRECTORY_NAME = 'segmotion'
EXPECTED_STATS_FILE_NAME_ZIPPED = (
    'segmotion/20170910/TrackingTable/scale_50000000m2/20170910-181300.xml.gz')
EXPECTED_POLYGON_FILE_NAME_ZIPPED = (
    'segmotion/20170910/ClusterID/scale_50000000m2/20170910-181300.netcdf.gz')
EXPECTED_STATS_FILE_NAME_UNZIPPED = (
    'segmotion/20170910/TrackingTable/scale_50000000m2/20170910-181300.xml')
EXPECTED_POLYGON_FILE_NAME_UNZIPPED = (
    'segmotion/20170910/ClusterID/scale_50000000m2/20170910-181300.netcdf')

TOP_PROCESSED_DIR_NAME = 'segmotion_processed'
EXPECTED_PROCESSED_FILE_NAME = (
    'segmotion_processed/20170910/scale_50000000m2/segmotion_'
    '2017-09-10-181300.p')

MIN_BUFFER_DISTS_METRES = numpy.array([numpy.nan, 0., 5000.])
MAX_BUFFER_DISTS_METRES = numpy.array([0., 5000., 10000.])
BUFFER_LAT_COLUMN_NAMES = ['vertex_latitudes_deg_buffer_0m',
                           'vertex_latitudes_deg_buffer_0_5000m',
                           'vertex_latitudes_deg_buffer_5000_10000m']
BUFFER_LNG_COLUMN_NAMES = ['vertex_longitudes_deg_buffer_0m',
                           'vertex_longitudes_deg_buffer_0_5000m',
                           'vertex_longitudes_deg_buffer_5000_10000m']

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

    def test_xml_column_name_orig_to_new(self):
        """Ensures correct output from _xml_column_name_orig_to_new."""

        this_xml_column_name = segmotion_io._xml_column_name_orig_to_new(
            XML_COLUMN_NAME_ORIG)
        self.assertTrue(this_xml_column_name == XML_COLUMN_NAME)

    def test_remove_rows_with_nan(self):
        """Ensures correct output from _remove_rows_with_nan."""

        this_stats_table = segmotion_io._remove_rows_with_nan(
            STATS_TABLE_WITH_NANS)
        self.assertTrue(this_stats_table.equals(STATS_TABLE_WITHOUT_NANS))

    def test_append_spc_date_to_storm_ids(self):
        """Ensures correct output from _append_spc_date_to_storm_ids."""

        these_storm_ids = segmotion_io._append_spc_date_to_storm_ids(
            STORM_IDS_NO_SPC_DATE, SPC_DATE_STRING)
        self.assertTrue(these_storm_ids == STORM_IDS_WITH_SPC_DATE)

    def test_storm_id_matrix_to_coord_lists(self):
        """Ensures correct output from _storm_id_matrix_to_coord_lists."""

        mask_table = segmotion_io._storm_id_matrix_to_coord_lists(
            NUMERIC_STORM_ID_MATRIX, UNIQUE_CENTER_LAT_DEG,
            UNIQUE_CENTER_LNG_DEG)

        self.assertTrue(numpy.array_equal(
            mask_table[STORM_ID_COLUMN].values,
            EXPECTED_POLYGON_TABLE[STORM_ID_COLUMN].values))

        num_storms = len(mask_table.index)
        for i in range(num_storms):
            self.assertTrue(numpy.array_equal(
                mask_table[GRID_POINT_ROW_COLUMN].values[i],
                EXPECTED_POLYGON_TABLE[GRID_POINT_ROW_COLUMN].values[i]))
            self.assertTrue(numpy.array_equal(
                mask_table[GRID_POINT_COLUMN_COLUMN].values[i],
                EXPECTED_POLYGON_TABLE[GRID_POINT_COLUMN_COLUMN].values[i]))
            self.assertTrue(numpy.allclose(
                mask_table[GRID_POINT_LAT_COLUMN].values[i],
                EXPECTED_POLYGON_TABLE[GRID_POINT_LAT_COLUMN].values[i],
                atol=TOLERANCE))
            self.assertTrue(numpy.allclose(
                mask_table[GRID_POINT_LNG_COLUMN].values[i],
                EXPECTED_POLYGON_TABLE[GRID_POINT_LNG_COLUMN].values[i],
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

    def test_get_pathless_stats_file_name_zipped(self):
        """Ensures correct output from _get_pathless_stats_file_name.

        In this case the expected file is zipped.
        """

        this_pathless_file_name = segmotion_io._get_pathless_stats_file_name(
            UNIX_TIME_SEC, zipped=True)
        self.assertTrue(
            this_pathless_file_name == PATHLESS_STATS_FILE_NAME_ZIPPED)

    def test_get_pathless_polygon_file_name_zipped(self):
        """Ensures correct output from _get_pathless_polygon_file_name.

        In this case the expected file is zipped.
        """

        this_pathless_file_name = segmotion_io._get_pathless_polygon_file_name(
            UNIX_TIME_SEC, zipped=True)
        self.assertTrue(
            this_pathless_file_name == PATHLESS_POLYGON_FILE_NAME_ZIPPED)

    def test_get_pathless_stats_file_name_unzipped(self):
        """Ensures correct output from _get_pathless_stats_file_name.

        In this case the expected file is unzipped.
        """

        this_pathless_file_name = segmotion_io._get_pathless_stats_file_name(
            UNIX_TIME_SEC, zipped=False)
        self.assertTrue(
            this_pathless_file_name == PATHLESS_STATS_FILE_NAME_UNZIPPED)

    def test_get_pathless_polygon_file_name_unzipped(self):
        """Ensures correct output from _get_pathless_polygon_file_name.

        In this case the expected file is unzipped.
        """

        this_pathless_file_name = segmotion_io._get_pathless_polygon_file_name(
            UNIX_TIME_SEC, zipped=False)
        self.assertTrue(
            this_pathless_file_name == PATHLESS_POLYGON_FILE_NAME_UNZIPPED)

    def test_get_pathless_processed_file_name(self):
        """Ensures correct output from _get_pathless_processed_file_name."""

        this_pathless_file_name = (
            segmotion_io._get_pathless_processed_file_name(UNIX_TIME_SEC))
        self.assertTrue(this_pathless_file_name == PATHLESS_PROCESSED_FILE_NAME)

    def test_get_relative_stats_dir_ordinal_scale(self):
        """Ensures correct output from _get_relative_stats_dir_ordinal_scale."""

        this_relative_dir_name = (
            segmotion_io._get_relative_stats_dir_ordinal_scale(
                SPC_DATE_STRING, TRACKING_SCALE_ORDINAL))
        self.assertTrue(
            this_relative_dir_name == RELATIVE_STATS_DIR_NAME_ORDINAL_SCALE)

    def test_get_relative_polygon_dir_ordinal_scale(self):
        """Need correct output from _get_relative_polygon_dir_ordinal_scale."""

        this_relative_dir_name = (
            segmotion_io._get_relative_polygon_dir_ordinal_scale(
                SPC_DATE_STRING, TRACKING_SCALE_ORDINAL))
        self.assertTrue(
            this_relative_dir_name == RELATIVE_POLYGON_DIR_NAME_ORDINAL_SCALE)

    def test_get_relative_stats_dir_physical_scale(self):
        """Need correct output from _get_relative_stats_dir_physical_scale."""

        this_relative_dir_name = (
            segmotion_io._get_relative_stats_dir_physical_scale(
                SPC_DATE_STRING, TRACKING_SCALE_METRES2))
        self.assertTrue(
            this_relative_dir_name == RELATIVE_STATS_DIR_NAME_PHYSICAL_SCALE)

    def test_get_relative_polygon_dir_physical_scale(self):
        """Need correct output from _get_relative_polygon_dir_physical_scale."""

        this_relative_dir_name = (
            segmotion_io._get_relative_polygon_dir_physical_scale(
                SPC_DATE_STRING, TRACKING_SCALE_METRES2))
        self.assertTrue(
            this_relative_dir_name == RELATIVE_POLYGON_DIR_NAME_PHYSICAL_SCALE)

    def test_get_relative_processed_directory(self):
        """Ensures correct output from _get_relative_processed_directory."""

        this_relative_dir_name = (
            segmotion_io._get_relative_processed_directory(
                SPC_DATE_STRING, TRACKING_SCALE_METRES2))
        self.assertTrue(this_relative_dir_name == RELATIVE_PROCESSED_DIR_NAME)

    def test_find_local_stats_file_zipped(self):
        """Ensures correct output from find_local_stats_file.

        In this case the expected file is zipped.
        """

        this_stats_file_name = segmotion_io.find_local_stats_file(
            unix_time_sec=UNIX_TIME_SEC, spc_date_unix_sec=SPC_DATE_UNIX_SEC,
            top_raw_directory_name=TOP_RAW_DIRECTORY_NAME,
            tracking_scale_metres2=TRACKING_SCALE_METRES2, zipped=True,
            raise_error_if_missing=False)

        self.assertTrue(this_stats_file_name == EXPECTED_STATS_FILE_NAME_ZIPPED)

    def test_find_local_polygon_file_zipped(self):
        """Ensures correct output from find_local_polygon_file.

        In this case the expected file is zipped.
        """

        this_polygon_file_name = segmotion_io.find_local_polygon_file(
            unix_time_sec=UNIX_TIME_SEC, spc_date_unix_sec=SPC_DATE_UNIX_SEC,
            top_raw_directory_name=TOP_RAW_DIRECTORY_NAME,
            tracking_scale_metres2=TRACKING_SCALE_METRES2, zipped=True,
            raise_error_if_missing=False)

        self.assertTrue(
            this_polygon_file_name == EXPECTED_POLYGON_FILE_NAME_ZIPPED)

    def test_find_local_stats_file_unzipped(self):
        """Ensures correct output from find_local_stats_file.

        In this case the expected file is unzipped.
        """

        this_stats_file_name = segmotion_io.find_local_stats_file(
            unix_time_sec=UNIX_TIME_SEC, spc_date_unix_sec=SPC_DATE_UNIX_SEC,
            top_raw_directory_name=TOP_RAW_DIRECTORY_NAME,
            tracking_scale_metres2=TRACKING_SCALE_METRES2, zipped=False,
            raise_error_if_missing=False)

        self.assertTrue(
            this_stats_file_name == EXPECTED_STATS_FILE_NAME_UNZIPPED)

    def test_find_local_polygon_file_unzipped(self):
        """Ensures correct output from find_local_polygon_file.

        In this case the expected file is unzipped.
        """

        this_polygon_file_name = segmotion_io.find_local_polygon_file(
            unix_time_sec=UNIX_TIME_SEC, spc_date_unix_sec=SPC_DATE_UNIX_SEC,
            top_raw_directory_name=TOP_RAW_DIRECTORY_NAME,
            tracking_scale_metres2=TRACKING_SCALE_METRES2, zipped=False,
            raise_error_if_missing=False)

        self.assertTrue(
            this_polygon_file_name == EXPECTED_POLYGON_FILE_NAME_UNZIPPED)

    def test_find_processed_file(self):
        """Ensures correct output from find_processed_file."""

        this_processed_file_name = segmotion_io.find_processed_file(
            unix_time_sec=UNIX_TIME_SEC, spc_date_unix_sec=SPC_DATE_UNIX_SEC,
            top_processed_dir_name=TOP_PROCESSED_DIR_NAME,
            tracking_scale_metres2=TRACKING_SCALE_METRES2,
            raise_error_if_missing=False)

        self.assertTrue(
            this_processed_file_name == EXPECTED_PROCESSED_FILE_NAME)

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
            self.assertTrue(numpy.array_equal(
                joined_table[GRID_POINT_ROW_COLUMN].values[i],
                EXPECTED_JOINED_TABLE[GRID_POINT_ROW_COLUMN].values[i]))
            self.assertTrue(numpy.array_equal(
                joined_table[GRID_POINT_COLUMN_COLUMN].values[i],
                EXPECTED_JOINED_TABLE[GRID_POINT_COLUMN_COLUMN].values[i]))
            self.assertTrue(numpy.allclose(
                joined_table[GRID_POINT_LAT_COLUMN].values[i],
                EXPECTED_JOINED_TABLE[GRID_POINT_LAT_COLUMN].values[i],
                atol=TOLERANCE))
            self.assertTrue(numpy.allclose(
                joined_table[GRID_POINT_LNG_COLUMN].values[i],
                EXPECTED_JOINED_TABLE[GRID_POINT_LNG_COLUMN].values[i],
                atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
