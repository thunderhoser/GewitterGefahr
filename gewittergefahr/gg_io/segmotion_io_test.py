"""Unit tests for segmotion_io.py."""

import unittest
import numpy
import pandas
from gewittergefahr.gg_io import segmotion_io
from gewittergefahr.gg_io import storm_tracking_io as tracking_io

TOLERANCE = 1e-6

XML_COLUMN_NAME = tracking_io.NORTH_VELOCITY_COLUMN
XML_COLUMN_NAME_ORIG = segmotion_io.NORTH_VELOCITY_COLUMN_ORIG
STATS_COLUMN_NAMES = [
    tracking_io.EAST_VELOCITY_COLUMN, tracking_io.NORTH_VELOCITY_COLUMN,
    tracking_io.AGE_COLUMN]

SPC_DATE_STRING = '20170910'
SPC_DATE_UNIX_SEC = 1505066400
STORM_IDS_NO_SPC_DATE = ['989', '657', '212']
STORM_IDS_WITH_SPC_DATE = ['989_20170910', '657_20170910', '212_20170910']

UNIX_TIME_SEC = 1505067180  # 181300 UTC 10 Sep 2017
PATHLESS_STATS_FILE_NAME_ZIPPED = '20170910-181300.xml.gz'
PATHLESS_POLYGON_FILE_NAME_ZIPPED = '20170910-181300.netcdf.gz'
PATHLESS_STATS_FILE_NAME_UNZIPPED = '20170910-181300.xml'
PATHLESS_POLYGON_FILE_NAME_UNZIPPED = '20170910-181300.netcdf'

TRACKING_SCALE_ORDINAL = 0
TRACKING_SCALE_METRES2 = 5e7
RELATIVE_STATS_DIR_NAME_ORDINAL_SCALE = '20170910/PolygonTable/scale_0'
RELATIVE_POLYGON_DIR_NAME_ORDINAL_SCALE = '20170910/ClusterID/scale_0'
RELATIVE_STATS_DIR_NAME_PHYSICAL_SCALE = (
    '20170910/PolygonTable/scale_50000000m2')
RELATIVE_POLYGON_DIR_NAME_PHYSICAL_SCALE = '20170910/ClusterID/scale_50000000m2'

TOP_RAW_DIRECTORY_NAME = 'segmotion'
STATS_FILE_NAME_ZIPPED = (
    'segmotion/20170910/PolygonTable/scale_50000000m2/20170910-181300.xml.gz')
POLYGON_FILE_NAME_ZIPPED = (
    'segmotion/20170910/ClusterID/scale_50000000m2/20170910-181300.netcdf.gz')
STATS_FILE_NAME_UNZIPPED = (
    'segmotion/20170910/PolygonTable/scale_50000000m2/20170910-181300.xml')
POLYGON_FILE_NAME_UNZIPPED = (
    'segmotion/20170910/ClusterID/scale_50000000m2/20170910-181300.netcdf')

STORM_IDS = ['0', '1', '2', '3']
EAST_VELOCITIES_M_S01 = numpy.array([5., 7.5, 10., 8.])
NORTH_VELOCITIES_M_S01 = numpy.array([6., 9., numpy.nan, 3.])
STORM_AGES_SEC = numpy.array([3000, numpy.nan, 1000, 2700])

STATS_DICT = {tracking_io.STORM_ID_COLUMN: STORM_IDS,
              tracking_io.EAST_VELOCITY_COLUMN: EAST_VELOCITIES_M_S01,
              tracking_io.NORTH_VELOCITY_COLUMN: NORTH_VELOCITIES_M_S01,
              tracking_io.AGE_COLUMN: STORM_AGES_SEC}
STATS_TABLE = pandas.DataFrame.from_dict(STATS_DICT)

NUMERIC_STORM_ID_MATRIX = numpy.array(
    [[0, 0, 0, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan],
     [numpy.nan, 0, 0, 0, numpy.nan, numpy.nan, 1, 1],
     [numpy.nan, numpy.nan, 0, 0, numpy.nan, 1, 1, 1],
     [numpy.nan, numpy.nan, numpy.nan, numpy.nan, 1, 1, 1, 1],
     [numpy.nan, 2, 2, numpy.nan, 1, numpy.nan, numpy.nan, numpy.nan],
     [numpy.nan, 2, 2, 2, numpy.nan, numpy.nan, numpy.nan, 3]])

UNIQUE_STORM_IDS = ['0', '1', '2', '3']
POLYGON_DICT = {tracking_io.STORM_ID_COLUMN: UNIQUE_STORM_IDS}
POLYGON_TABLE = pandas.DataFrame.from_dict(POLYGON_DICT)

NESTED_ARRAY = POLYGON_TABLE[[
    tracking_io.STORM_ID_COLUMN, tracking_io.STORM_ID_COLUMN]].values.tolist()
ARGUMENT_DICT = {tracking_io.GRID_POINT_LAT_COLUMN: NESTED_ARRAY,
                 tracking_io.GRID_POINT_LNG_COLUMN: NESTED_ARRAY,
                 tracking_io.GRID_POINT_ROW_COLUMN: NESTED_ARRAY,
                 tracking_io.GRID_POINT_COLUMN_COLUMN: NESTED_ARRAY}
POLYGON_TABLE = POLYGON_TABLE.assign(**ARGUMENT_DICT)

POLYGON_TABLE[tracking_io.GRID_POINT_ROW_COLUMN].values[0] = numpy.array(
    [0, 0, 0, 1, 1, 1, 2, 2])
POLYGON_TABLE[tracking_io.GRID_POINT_COLUMN_COLUMN].values[0] = numpy.array(
    [0, 1, 2, 1, 2, 3, 2, 3])
POLYGON_TABLE[tracking_io.GRID_POINT_LAT_COLUMN].values[0] = numpy.array(
    [35., 35., 35., 35.1, 35.1, 35.1, 35.2, 35.2])
POLYGON_TABLE[tracking_io.GRID_POINT_LNG_COLUMN].values[0] = numpy.array(
    [262.7, 262.8, 262.9, 262.8, 262.9, 263., 262.9, 263.])

POLYGON_TABLE[tracking_io.GRID_POINT_ROW_COLUMN].values[1] = numpy.array(
    [1, 1, 2, 2, 2, 3, 3, 3, 3, 4])
POLYGON_TABLE[tracking_io.GRID_POINT_COLUMN_COLUMN].values[1] = numpy.array(
    [6, 7, 5, 6, 7, 4, 5, 6, 7, 4])
POLYGON_TABLE[tracking_io.GRID_POINT_LAT_COLUMN].values[1] = numpy.array(
    [35.1, 35.1, 35.2, 35.2, 35.2, 35.3, 35.3, 35.3, 35.3, 35.4])
POLYGON_TABLE[tracking_io.GRID_POINT_LNG_COLUMN].values[1] = numpy.array(
    [263.3, 263.4, 263.2, 263.3, 263.4, 263.1, 263.2, 263.3, 263.4, 263.1])

POLYGON_TABLE[tracking_io.GRID_POINT_ROW_COLUMN].values[2] = numpy.array(
    [4, 4, 5, 5, 5])
POLYGON_TABLE[tracking_io.GRID_POINT_COLUMN_COLUMN].values[2] = numpy.array(
    [1, 2, 1, 2, 3])
POLYGON_TABLE[tracking_io.GRID_POINT_LAT_COLUMN].values[2] = numpy.array(
    [35.4, 35.4, 35.5, 35.5, 35.5])
POLYGON_TABLE[tracking_io.GRID_POINT_LNG_COLUMN].values[2] = numpy.array(
    [262.8, 262.9, 262.8, 262.9, 263.])

POLYGON_TABLE[tracking_io.GRID_POINT_ROW_COLUMN].values[3] = numpy.array([5])
POLYGON_TABLE[tracking_io.GRID_POINT_COLUMN_COLUMN].values[3] = numpy.array([7])
POLYGON_TABLE[tracking_io.GRID_POINT_LAT_COLUMN].values[3] = numpy.array([35.5])
POLYGON_TABLE[tracking_io.GRID_POINT_LNG_COLUMN].values[3] = numpy.array(
    [263.4])

ARGUMENT_DICT = {tracking_io.EAST_VELOCITY_COLUMN: EAST_VELOCITIES_M_S01,
                 tracking_io.NORTH_VELOCITY_COLUMN: NORTH_VELOCITIES_M_S01,
                 tracking_io.AGE_COLUMN: STORM_AGES_SEC}
STATS_AND_POLYGON_TABLE = POLYGON_TABLE.assign(**ARGUMENT_DICT)


class SegmotionIoTests(unittest.TestCase):
    """Each method is a unit test for segmotion_io.py."""

    def test_xml_column_name_orig_to_new(self):
        """Ensures correct output from _xml_column_name_orig_to_new."""

        this_xml_column_name = segmotion_io._xml_column_name_orig_to_new(
            XML_COLUMN_NAME_ORIG)
        self.assertTrue(this_xml_column_name == XML_COLUMN_NAME)

    def test_append_spc_date_to_storm_ids(self):
        """Ensures correct output from _append_spc_date_to_storm_ids."""

        these_storm_ids = segmotion_io._append_spc_date_to_storm_ids(
            STORM_IDS_NO_SPC_DATE, SPC_DATE_STRING)
        self.assertTrue(these_storm_ids == STORM_IDS_WITH_SPC_DATE)

    def test_storm_id_matrix_to_coord_lists(self):
        """Ensures correct output from _storm_id_matrix_to_coord_lists."""

        this_polygon_table = segmotion_io._storm_id_matrix_to_coord_lists(
            NUMERIC_STORM_ID_MATRIX)

        self.assertTrue(numpy.array_equal(
            this_polygon_table[tracking_io.STORM_ID_COLUMN].values,
            POLYGON_TABLE[tracking_io.STORM_ID_COLUMN].values))

        num_storms = len(this_polygon_table.index)
        for i in range(num_storms):
            self.assertTrue(numpy.array_equal(
                this_polygon_table[tracking_io.GRID_POINT_ROW_COLUMN].values[i],
                POLYGON_TABLE[tracking_io.GRID_POINT_ROW_COLUMN].values[i]))
            self.assertTrue(numpy.array_equal(
                this_polygon_table[
                    tracking_io.GRID_POINT_COLUMN_COLUMN].values[i],
                POLYGON_TABLE[tracking_io.GRID_POINT_COLUMN_COLUMN].values[i]))

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

    def test_find_local_stats_file_zipped(self):
        """Ensures correct output from find_local_stats_file.

        In this case the expected file is zipped.
        """

        this_stats_file_name = segmotion_io.find_local_stats_file(
            unix_time_sec=UNIX_TIME_SEC, spc_date_unix_sec=SPC_DATE_UNIX_SEC,
            top_raw_directory_name=TOP_RAW_DIRECTORY_NAME,
            tracking_scale_metres2=TRACKING_SCALE_METRES2, zipped=True,
            raise_error_if_missing=False)

        self.assertTrue(this_stats_file_name == STATS_FILE_NAME_ZIPPED)

    def test_find_local_polygon_file_zipped(self):
        """Ensures correct output from find_local_polygon_file.

        In this case the expected file is zipped.
        """

        this_polygon_file_name = segmotion_io.find_local_polygon_file(
            unix_time_sec=UNIX_TIME_SEC, spc_date_unix_sec=SPC_DATE_UNIX_SEC,
            top_raw_directory_name=TOP_RAW_DIRECTORY_NAME,
            tracking_scale_metres2=TRACKING_SCALE_METRES2, zipped=True,
            raise_error_if_missing=False)

        self.assertTrue(this_polygon_file_name == POLYGON_FILE_NAME_ZIPPED)

    def test_find_local_stats_file_unzipped(self):
        """Ensures correct output from find_local_stats_file.

        In this case the expected file is unzipped.
        """

        this_stats_file_name = segmotion_io.find_local_stats_file(
            unix_time_sec=UNIX_TIME_SEC, spc_date_unix_sec=SPC_DATE_UNIX_SEC,
            top_raw_directory_name=TOP_RAW_DIRECTORY_NAME,
            tracking_scale_metres2=TRACKING_SCALE_METRES2, zipped=False,
            raise_error_if_missing=False)

        self.assertTrue(this_stats_file_name == STATS_FILE_NAME_UNZIPPED)

    def test_find_local_polygon_file_unzipped(self):
        """Ensures correct output from find_local_polygon_file.

        In this case the expected file is unzipped.
        """

        this_polygon_file_name = segmotion_io.find_local_polygon_file(
            unix_time_sec=UNIX_TIME_SEC, spc_date_unix_sec=SPC_DATE_UNIX_SEC,
            top_raw_directory_name=TOP_RAW_DIRECTORY_NAME,
            tracking_scale_metres2=TRACKING_SCALE_METRES2, zipped=False,
            raise_error_if_missing=False)

        self.assertTrue(this_polygon_file_name == POLYGON_FILE_NAME_UNZIPPED)

    def test_join_stats_and_polygons(self):
        """Ensures correct output from join_stats_and_polygons."""

        this_stats_and_polygon_table = segmotion_io.join_stats_and_polygons(
            POLYGON_TABLE, STATS_TABLE)

        this_stats_table = this_stats_and_polygon_table[STATS_COLUMN_NAMES]
        expected_stats_table = STATS_AND_POLYGON_TABLE[STATS_COLUMN_NAMES]
        self.assertTrue(this_stats_table.equals(expected_stats_table))

        num_storms = len(this_stats_and_polygon_table.index)
        for i in range(num_storms):
            self.assertTrue(numpy.array_equal(
                this_stats_and_polygon_table[
                    tracking_io.GRID_POINT_ROW_COLUMN].values[i],
                STATS_AND_POLYGON_TABLE[
                    tracking_io.GRID_POINT_ROW_COLUMN].values[i]))

            self.assertTrue(numpy.array_equal(
                this_stats_and_polygon_table[
                    tracking_io.GRID_POINT_COLUMN_COLUMN].values[i],
                STATS_AND_POLYGON_TABLE[
                    tracking_io.GRID_POINT_COLUMN_COLUMN].values[i]))

            self.assertTrue(numpy.allclose(
                this_stats_and_polygon_table[
                    tracking_io.GRID_POINT_LAT_COLUMN].values[i],
                STATS_AND_POLYGON_TABLE[
                    tracking_io.GRID_POINT_LAT_COLUMN].values[i],
                atol=TOLERANCE))

            self.assertTrue(numpy.allclose(
                this_stats_and_polygon_table[
                    tracking_io.GRID_POINT_LNG_COLUMN].values[i],
                STATS_AND_POLYGON_TABLE[
                    tracking_io.GRID_POINT_LNG_COLUMN].values[i],
                atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
