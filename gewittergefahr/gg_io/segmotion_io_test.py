"""Unit tests for segmotion_io.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_io import segmotion_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

TOLERANCE = 1e-6

XML_COLUMN_NAME = tracking_utils.NORTH_VELOCITY_COLUMN
XML_COLUMN_NAME_ORIG = segmotion_io.NORTH_VELOCITY_COLUMN_ORIG
STATS_COLUMN_NAMES = [
    tracking_utils.EAST_VELOCITY_COLUMN, tracking_utils.NORTH_VELOCITY_COLUMN,
    tracking_utils.AGE_COLUMN
]

SPC_DATE_STRING = '20170910'
ID_STRINGS_NO_DATE = ['989', '657', '212']
ID_STRINGS_WITH_DATE = ['989-20170910', '657-20170910', '212-20170910']

UNIX_TIME_SEC = 1505067180  # 181300 UTC 10 Sep 2017
PATHLESS_STATS_FILE_NAME_ZIPPED = '20170910-181300.xml.gz'
PATHLESS_POLYGON_FILE_NAME_ZIPPED = '20170910-181300.netcdf.gz'
PATHLESS_STATS_FILE_NAME_UNZIPPED = '20170910-181300.xml'
PATHLESS_POLYGON_FILE_NAME_UNZIPPED = '20170910-181300.netcdf'

TRACKING_SCALE_ORDINAL = 0
TRACKING_SCALE_METRES2 = 5e7
RELATIVE_STATS_DIR_NAME_ORDINAL_SCALE = 'PolygonTable/scale_0'
RELATIVE_POLYGON_DIR_NAME_ORDINAL_SCALE = 'ClusterID/scale_0'
RELATIVE_STATS_DIR_NAME_PHYSICAL_SCALE = 'PolygonTable/scale_50000000m2'
RELATIVE_POLYGON_DIR_NAME_PHYSICAL_SCALE = 'ClusterID/scale_50000000m2'

TOP_RAW_DIRECTORY_NAME = 'segmotion'
STATS_FILE_NAME_ZIPPED = (
    'segmotion/2017/20170910/PolygonTable/scale_50000000m2/'
    '20170910-181300.xml.gz'
)
POLYGON_FILE_NAME_ZIPPED = (
    'segmotion/2017/20170910/ClusterID/scale_50000000m2/'
    '20170910-181300.netcdf.gz'
)

PRIMARY_ID_STRINGS = ['0', '1', '2', '3']
EAST_VELOCITIES_M_S01 = numpy.array([5, 7.5, 10, 8])
NORTH_VELOCITIES_M_S01 = numpy.array([6, 9, numpy.nan, 3])
STORM_AGES_SEC = numpy.array([3000, numpy.nan, 1000, 2700])

STATS_DICT = {
    tracking_utils.PRIMARY_ID_COLUMN: PRIMARY_ID_STRINGS,
    tracking_utils.EAST_VELOCITY_COLUMN: EAST_VELOCITIES_M_S01,
    tracking_utils.NORTH_VELOCITY_COLUMN: NORTH_VELOCITIES_M_S01,
    tracking_utils.AGE_COLUMN: STORM_AGES_SEC
}

STATS_TABLE = pandas.DataFrame.from_dict(STATS_DICT)

numeric_id_matrix = numpy.array([
    [0, 0, 0, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan],
    [numpy.nan, 0, 0, 0, numpy.nan, numpy.nan, 1, 1],
    [numpy.nan, numpy.nan, 0, 0, numpy.nan, 1, 1, 1],
    [numpy.nan, numpy.nan, numpy.nan, numpy.nan, 1, 1, 1, 1],
    [numpy.nan, 2, 2, numpy.nan, 1, numpy.nan, numpy.nan, numpy.nan],
    [numpy.nan, 2, 2, 2, numpy.nan, numpy.nan, numpy.nan, 3]
])

UNIQUE_ID_STRINGS = ['0', '1', '2', '3']
POLYGON_DICT = {tracking_utils.PRIMARY_ID_COLUMN: UNIQUE_ID_STRINGS}
POLYGON_TABLE = pandas.DataFrame.from_dict(POLYGON_DICT)

NESTED_ARRAY = POLYGON_TABLE[[
    tracking_utils.PRIMARY_ID_COLUMN, tracking_utils.PRIMARY_ID_COLUMN
]].values.tolist()

ARGUMENT_DICT = {
    tracking_utils.LATITUDES_IN_STORM_COLUMN: NESTED_ARRAY,
    tracking_utils.LONGITUDES_IN_STORM_COLUMN: NESTED_ARRAY,
    tracking_utils.ROWS_IN_STORM_COLUMN: NESTED_ARRAY,
    tracking_utils.COLUMNS_IN_STORM_COLUMN: NESTED_ARRAY
}

POLYGON_TABLE = POLYGON_TABLE.assign(**ARGUMENT_DICT)

POLYGON_TABLE[tracking_utils.ROWS_IN_STORM_COLUMN].values[0] = numpy.array(
    [0, 0, 0, 1, 1, 1, 2, 2], dtype=int
)
POLYGON_TABLE[tracking_utils.COLUMNS_IN_STORM_COLUMN].values[0] = numpy.array(
    [0, 1, 2, 1, 2, 3, 2, 3], dtype=int
)
POLYGON_TABLE[tracking_utils.LATITUDES_IN_STORM_COLUMN].values[0] = numpy.array(
    [35, 35, 35, 35.1, 35.1, 35.1, 35.2, 35.2]
)
POLYGON_TABLE[tracking_utils.LONGITUDES_IN_STORM_COLUMN].values[0] = (
    numpy.array([262.7, 262.8, 262.9, 262.8, 262.9, 263, 262.9, 263])
)

POLYGON_TABLE[tracking_utils.ROWS_IN_STORM_COLUMN].values[1] = numpy.array(
    [1, 1, 2, 2, 2, 3, 3, 3, 3, 4], dtype=int
)
POLYGON_TABLE[tracking_utils.COLUMNS_IN_STORM_COLUMN].values[1] = numpy.array(
    [6, 7, 5, 6, 7, 4, 5, 6, 7, 4], dtype=int
)
POLYGON_TABLE[tracking_utils.LATITUDES_IN_STORM_COLUMN].values[1] = numpy.array(
    [35.1, 35.1, 35.2, 35.2, 35.2, 35.3, 35.3, 35.3, 35.3, 35.4]
)
POLYGON_TABLE[tracking_utils.LONGITUDES_IN_STORM_COLUMN].values[1] = (
    numpy.array(
        [263.3, 263.4, 263.2, 263.3, 263.4, 263.1, 263.2, 263.3, 263.4, 263.1]
    )
)

POLYGON_TABLE[tracking_utils.ROWS_IN_STORM_COLUMN].values[2] = numpy.array(
    [4, 4, 5, 5, 5])
POLYGON_TABLE[tracking_utils.COLUMNS_IN_STORM_COLUMN].values[2] = numpy.array(
    [1, 2, 1, 2, 3])
POLYGON_TABLE[tracking_utils.LATITUDES_IN_STORM_COLUMN].values[2] = numpy.array(
    [35.4, 35.4, 35.5, 35.5, 35.5])
POLYGON_TABLE[tracking_utils.LONGITUDES_IN_STORM_COLUMN].values[2] = numpy.array(
    [262.8, 262.9, 262.8, 262.9, 263])

POLYGON_TABLE[tracking_utils.ROWS_IN_STORM_COLUMN].values[3] = numpy.array(
    [5], dtype=int
)
POLYGON_TABLE[tracking_utils.COLUMNS_IN_STORM_COLUMN].values[3] = numpy.array(
    [7], dtype=int
)
POLYGON_TABLE[tracking_utils.LATITUDES_IN_STORM_COLUMN].values[3] = numpy.array(
    [35.5]
)
POLYGON_TABLE[tracking_utils.LONGITUDES_IN_STORM_COLUMN].values[3] = (
    numpy.array([263.4])
)

ARGUMENT_DICT = {
    tracking_utils.EAST_VELOCITY_COLUMN: EAST_VELOCITIES_M_S01,
    tracking_utils.NORTH_VELOCITY_COLUMN: NORTH_VELOCITIES_M_S01,
    tracking_utils.AGE_COLUMN: STORM_AGES_SEC
}

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

        these_id_strings = segmotion_io._append_spc_date_to_storm_ids(
            primary_id_strings=ID_STRINGS_NO_DATE,
            spc_date_string=SPC_DATE_STRING)

        self.assertTrue(these_id_strings == ID_STRINGS_WITH_DATE)

    def test_id_matrix_to_coord_lists(self):
        """Ensures correct output from _id_matrix_to_coord_lists."""

        this_polygon_table = segmotion_io._id_matrix_to_coord_lists(
            numeric_id_matrix)

        self.assertTrue(numpy.array_equal(
            this_polygon_table[tracking_utils.PRIMARY_ID_COLUMN].values,
            POLYGON_TABLE[tracking_utils.PRIMARY_ID_COLUMN].values
        ))

        num_storms = len(this_polygon_table.index)

        for i in range(num_storms):
            self.assertTrue(numpy.array_equal(
                this_polygon_table[
                    tracking_utils.ROWS_IN_STORM_COLUMN].values[i],
                POLYGON_TABLE[tracking_utils.ROWS_IN_STORM_COLUMN].values[i]
            ))

            self.assertTrue(numpy.array_equal(
                this_polygon_table[
                    tracking_utils.COLUMNS_IN_STORM_COLUMN].values[i],
                POLYGON_TABLE[tracking_utils.COLUMNS_IN_STORM_COLUMN].values[i]
            ))

    def test_get_pathless_stats_file_name_zipped(self):
        """Ensures correct output from _get_pathless_stats_file_name.

        In this case the expected file is zipped.
        """

        this_pathless_file_name = segmotion_io._get_pathless_stats_file_name(
            unix_time_sec=UNIX_TIME_SEC, zipped=True)

        self.assertTrue(
            this_pathless_file_name == PATHLESS_STATS_FILE_NAME_ZIPPED
        )

    def test_get_pathless_polygon_file_name_zipped(self):
        """Ensures correct output from _get_pathless_polygon_file_name.

        In this case the expected file is zipped.
        """

        this_pathless_file_name = segmotion_io._get_pathless_polygon_file_name(
            unix_time_sec=UNIX_TIME_SEC, zipped=True)

        self.assertTrue(
            this_pathless_file_name == PATHLESS_POLYGON_FILE_NAME_ZIPPED
        )

    def test_get_pathless_stats_file_name_unzipped(self):
        """Ensures correct output from _get_pathless_stats_file_name.

        In this case the expected file is unzipped.
        """

        this_pathless_file_name = segmotion_io._get_pathless_stats_file_name(
            unix_time_sec=UNIX_TIME_SEC, zipped=False)

        self.assertTrue(
            this_pathless_file_name == PATHLESS_STATS_FILE_NAME_UNZIPPED
        )

    def test_get_pathless_polygon_file_name_unzipped(self):
        """Ensures correct output from _get_pathless_polygon_file_name.

        In this case the expected file is unzipped.
        """

        this_pathless_file_name = segmotion_io._get_pathless_polygon_file_name(
            unix_time_sec=UNIX_TIME_SEC, zipped=False)

        self.assertTrue(
            this_pathless_file_name == PATHLESS_POLYGON_FILE_NAME_UNZIPPED
        )

    def test_get_relative_stats_dir_ordinal_scale(self):
        """Ensures correct output from _get_relative_stats_dir_ordinal_scale."""

        this_relative_dir_name = (
            segmotion_io._get_relative_stats_dir_ordinal_scale(
                TRACKING_SCALE_ORDINAL)
        )

        self.assertTrue(
            this_relative_dir_name == RELATIVE_STATS_DIR_NAME_ORDINAL_SCALE
        )

    def test_get_relative_polygon_dir_ordinal_scale(self):
        """Need correct output from _get_relative_polygon_dir_ordinal_scale."""

        this_relative_dir_name = (
            segmotion_io._get_relative_polygon_dir_ordinal_scale(
                TRACKING_SCALE_ORDINAL)
        )

        self.assertTrue(
            this_relative_dir_name == RELATIVE_POLYGON_DIR_NAME_ORDINAL_SCALE
        )

    def test_get_relative_stats_dir_physical_scale(self):
        """Need correct output from _get_relative_stats_dir_physical_scale."""

        this_relative_dir_name = (
            segmotion_io._get_relative_stats_dir_physical_scale(
                TRACKING_SCALE_METRES2)
        )

        self.assertTrue(
            this_relative_dir_name == RELATIVE_STATS_DIR_NAME_PHYSICAL_SCALE
        )

    def test_get_relative_polygon_dir_physical_scale(self):
        """Need correct output from _get_relative_polygon_dir_physical_scale."""

        this_relative_dir_name = (
            segmotion_io._get_relative_polygon_dir_physical_scale(
                TRACKING_SCALE_METRES2)
        )

        self.assertTrue(
            this_relative_dir_name == RELATIVE_POLYGON_DIR_NAME_PHYSICAL_SCALE
        )

    def test_find_local_stats_file(self):
        """Ensures correct output from find_local_stats_file."""

        this_stats_file_name = segmotion_io.find_local_stats_file(
            unix_time_sec=UNIX_TIME_SEC, spc_date_string=SPC_DATE_STRING,
            top_raw_directory_name=TOP_RAW_DIRECTORY_NAME,
            tracking_scale_metres2=TRACKING_SCALE_METRES2,
            raise_error_if_missing=False)

        self.assertTrue(this_stats_file_name == STATS_FILE_NAME_ZIPPED)

    def test_find_local_polygon_file(self):
        """Ensures correct output from find_local_polygon_file."""

        this_polygon_file_name = segmotion_io.find_local_polygon_file(
            unix_time_sec=UNIX_TIME_SEC, spc_date_string=SPC_DATE_STRING,
            top_raw_directory_name=TOP_RAW_DIRECTORY_NAME,
            tracking_scale_metres2=TRACKING_SCALE_METRES2,
            raise_error_if_missing=False)

        self.assertTrue(this_polygon_file_name == POLYGON_FILE_NAME_ZIPPED)

    def test_join_stats_and_polygons(self):
        """Ensures correct output from join_stats_and_polygons."""

        this_stats_and_polygon_table = segmotion_io.join_stats_and_polygons(
            polygon_table=copy.deepcopy(POLYGON_TABLE),
            stats_table=copy.deepcopy(STATS_TABLE)
        )

        this_stats_table = this_stats_and_polygon_table[STATS_COLUMN_NAMES]
        expected_stats_table = STATS_AND_POLYGON_TABLE[STATS_COLUMN_NAMES]
        self.assertTrue(this_stats_table.equals(expected_stats_table))

        num_storms = len(this_stats_and_polygon_table.index)

        for i in range(num_storms):
            self.assertTrue(numpy.array_equal(
                this_stats_and_polygon_table[
                    tracking_utils.ROWS_IN_STORM_COLUMN].values[i],
                STATS_AND_POLYGON_TABLE[
                    tracking_utils.ROWS_IN_STORM_COLUMN].values[i]
            ))

            self.assertTrue(numpy.array_equal(
                this_stats_and_polygon_table[
                    tracking_utils.COLUMNS_IN_STORM_COLUMN].values[i],
                STATS_AND_POLYGON_TABLE[
                    tracking_utils.COLUMNS_IN_STORM_COLUMN].values[i]
            ))

            self.assertTrue(numpy.allclose(
                this_stats_and_polygon_table[
                    tracking_utils.LATITUDES_IN_STORM_COLUMN].values[i],
                STATS_AND_POLYGON_TABLE[
                    tracking_utils.LATITUDES_IN_STORM_COLUMN].values[i],
                atol=TOLERANCE
            ))

            self.assertTrue(numpy.allclose(
                this_stats_and_polygon_table[
                    tracking_utils.LONGITUDES_IN_STORM_COLUMN].values[i],
                STATS_AND_POLYGON_TABLE[
                    tracking_utils.LONGITUDES_IN_STORM_COLUMN].values[i],
                atol=TOLERANCE
            ))


if __name__ == '__main__':
    unittest.main()
