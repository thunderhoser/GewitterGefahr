"""Unit tests for myrorss_and_mrms_io.py."""

import unittest
import numpy
import pandas
from gewittergefahr.gg_io import myrorss_and_mrms_io
from gewittergefahr.gg_utils import radar_utils

TOLERANCE = 1e-6

LL_SHEAR_NAME_MYRORSS = radar_utils.LOW_LEVEL_SHEAR_NAME_MYRORSS
LL_SHEAR_NAME_MRMS = radar_utils.LOW_LEVEL_SHEAR_NAME_MRMS
LL_SHEAR_NAME_NEW = radar_utils.LOW_LEVEL_SHEAR_NAME

# The following constants are used to test _get_pathless_raw_file_pattern and
# _get_pathless_raw_file_name.
FILE_TIME_UNIX_SEC = 1507234802
FILE_SPC_DATE_STRING = '20171005'
PATHLESS_ZIPPED_FILE_NAME = '20171005-202002.netcdf.gz'
PATHLESS_UNZIPPED_FILE_NAME = '20171005-202002.netcdf'
PATHLESS_FILE_PATTERN = '20171005-2020*.netcdf*'

# The following constants are used to test _remove_sentinels_from_sparse_grid.
THESE_GRID_ROWS = numpy.linspace(0, 10, num=11, dtype=int)
THESE_GRID_COLUMNS = numpy.linspace(0, 10, num=11, dtype=int)
THESE_NUM_GRID_CELLS = numpy.linspace(0, 10, num=11, dtype=int)

SENTINEL_VALUES = numpy.array([-99000., -99001.])
RADAR_FIELD_WITH_SENTINELS = radar_utils.VIL_NAME
THESE_RADAR_VALUES = numpy.array(
    [SENTINEL_VALUES[0], 1., SENTINEL_VALUES[1], 3., SENTINEL_VALUES[0], 5.,
     SENTINEL_VALUES[1], 7., 8., 9., 10.])

THIS_DICTIONARY = {myrorss_and_mrms_io.GRID_ROW_COLUMN: THESE_GRID_ROWS,
                   myrorss_and_mrms_io.GRID_COLUMN_COLUMN: THESE_GRID_COLUMNS,
                   myrorss_and_mrms_io.NUM_GRID_CELL_COLUMN:
                       THESE_NUM_GRID_CELLS,
                   RADAR_FIELD_WITH_SENTINELS: THESE_RADAR_VALUES}
SPARSE_GRID_TABLE_WITH_SENTINELS = pandas.DataFrame.from_dict(THIS_DICTIONARY)

THESE_SENTINEL_INDICES = numpy.array([0, 2, 4, 6], dtype=int)
SPARSE_GRID_TABLE_NO_SENTINELS = SPARSE_GRID_TABLE_WITH_SENTINELS.drop(
    SPARSE_GRID_TABLE_WITH_SENTINELS.index[THESE_SENTINEL_INDICES], axis=0,
    inplace=False)

# The following constants are used to test _remove_sentinels_from_full_grid.
FIELD_MATRIX_WITH_SENTINELS = numpy.array([
    [0, 1, 2],
    [3, SENTINEL_VALUES[0], 5],
    [SENTINEL_VALUES[1], 7, 8],
    [9, 10, SENTINEL_VALUES[1]],
    [12, 13, SENTINEL_VALUES[0]]])
FIELD_MATRIX_NO_SENTINELS = numpy.array([
    [0, 1, 2],
    [3, numpy.nan, 5],
    [numpy.nan, 7, 8],
    [9, 10, numpy.nan],
    [12, 13, numpy.nan]])

# The following constants are used to test get_relative_dir_for_raw_files.
RELATIVE_DIR_NAME_MYRORSS = '{0:s}/00.25'.format(LL_SHEAR_NAME_MYRORSS)
RELATIVE_DIR_NAME_MRMS = '{0:s}/00.25'.format(LL_SHEAR_NAME_MRMS)

# The following constants are used to test find_raw_file and
# find_raw_file_inexact_time.
TOP_RAW_DIRECTORY_NAME = 'radar'
RAW_FILE_NAME_MYRORSS = (
    'radar/2017/20171005/{0:s}/00.25/20171005-202002.netcdf.gz'.format(
        LL_SHEAR_NAME_MYRORSS))
RAW_FILE_NAME_MRMS = (
    'radar/2017/20171005/{0:s}/00.25/20171005-202002.netcdf.gz'.format(
        LL_SHEAR_NAME_MRMS))


class MyrorssAndMrmsIoTests(unittest.TestCase):
    """Each method is a unit test for myrorss_and_mrms_io.py."""

    def test_get_pathless_raw_file_pattern(self):
        """Ensures correct output from _get_pathless_raw_file_pattern."""

        this_pathless_file_pattern = (
            myrorss_and_mrms_io._get_pathless_raw_file_pattern(
                FILE_TIME_UNIX_SEC))
        self.assertTrue(this_pathless_file_pattern == PATHLESS_FILE_PATTERN)

    def test_get_pathless_raw_file_name_zipped(self):
        """Ensures correct output from _get_pathless_raw_file_name.

        In this case, generating name for zipped file.
        """

        this_pathless_file_name = (
            myrorss_and_mrms_io._get_pathless_raw_file_name(
                FILE_TIME_UNIX_SEC, zipped=True))
        self.assertTrue(this_pathless_file_name == PATHLESS_ZIPPED_FILE_NAME)

    def test_get_pathless_raw_file_name_unzipped(self):
        """Ensures correct output from _get_pathless_raw_file_name.

        In this case, generating name for unzipped file.
        """

        this_pathless_file_name = (
            myrorss_and_mrms_io._get_pathless_raw_file_name(
                FILE_TIME_UNIX_SEC, zipped=False))
        self.assertTrue(this_pathless_file_name == PATHLESS_UNZIPPED_FILE_NAME)

    def test_raw_file_name_to_time_zipped(self):
        """Ensures correct output from _raw_file_name_to_time.

        In this case, input is name of zipped file.
        """

        this_time_unix_sec = myrorss_and_mrms_io._raw_file_name_to_time(
            PATHLESS_ZIPPED_FILE_NAME)
        self.assertTrue(this_time_unix_sec == FILE_TIME_UNIX_SEC)

    def test_raw_file_name_to_time_unzipped(self):
        """Ensures correct output from _raw_file_name_to_time.

        In this case, input is name of unzipped file.
        """

        this_time_unix_sec = myrorss_and_mrms_io._raw_file_name_to_time(
            PATHLESS_UNZIPPED_FILE_NAME)
        self.assertTrue(this_time_unix_sec == FILE_TIME_UNIX_SEC)

    def test_remove_sentinels_from_sparse_grid(self):
        """Ensures correct output from _remove_sentinels_from_sparse_grid."""

        this_sparse_grid_table = (
            myrorss_and_mrms_io._remove_sentinels_from_sparse_grid(
                SPARSE_GRID_TABLE_WITH_SENTINELS,
                field_name=RADAR_FIELD_WITH_SENTINELS,
                sentinel_values=SENTINEL_VALUES))
        self.assertTrue(
            this_sparse_grid_table.equals(SPARSE_GRID_TABLE_NO_SENTINELS))

    def test_remove_sentinels_from_full_grid(self):
        """Ensures correct output from _remove_sentinels_from_full_grid."""

        this_field_matrix = (
            myrorss_and_mrms_io._remove_sentinels_from_full_grid(
                FIELD_MATRIX_WITH_SENTINELS, SENTINEL_VALUES))
        self.assertTrue(numpy.allclose(
            this_field_matrix, FIELD_MATRIX_NO_SENTINELS, atol=TOLERANCE,
            equal_nan=True))

    def test_get_relative_dir_for_raw_files_myrorss(self):
        """Ensures correct output from get_relative_dir_for_raw_files.

        In this case, data source is MYRORSS.
        """

        this_relative_dir_name = (
            myrorss_and_mrms_io.get_relative_dir_for_raw_files(
                field_name=LL_SHEAR_NAME_NEW,
                data_source=radar_utils.MYRORSS_SOURCE_ID))
        self.assertTrue(this_relative_dir_name == RELATIVE_DIR_NAME_MYRORSS)

    def test_get_relative_dir_for_raw_files_mrms(self):
        """Ensures correct output from get_relative_dir_for_raw_files.

        In this case, data source is MRMS.
        """

        this_relative_dir_name = (
            myrorss_and_mrms_io.get_relative_dir_for_raw_files(
                field_name=LL_SHEAR_NAME_NEW,
                data_source=radar_utils.MRMS_SOURCE_ID))
        self.assertTrue(this_relative_dir_name == RELATIVE_DIR_NAME_MRMS)

    def test_find_raw_file_myrorss(self):
        """Ensures correct output from find_raw_file."""

        this_raw_file_name = myrorss_and_mrms_io.find_raw_file(
            unix_time_sec=FILE_TIME_UNIX_SEC,
            spc_date_string=FILE_SPC_DATE_STRING,
            field_name=LL_SHEAR_NAME_NEW,
            data_source=radar_utils.MYRORSS_SOURCE_ID,
            top_directory_name=TOP_RAW_DIRECTORY_NAME,
            raise_error_if_missing=False)
        self.assertTrue(this_raw_file_name == RAW_FILE_NAME_MYRORSS)

    def test_find_raw_file_mrms(self):
        """Ensures correct output from find_raw_file."""

        this_raw_file_name = myrorss_and_mrms_io.find_raw_file(
            unix_time_sec=FILE_TIME_UNIX_SEC,
            spc_date_string=FILE_SPC_DATE_STRING,
            field_name=LL_SHEAR_NAME_NEW,
            data_source=radar_utils.MRMS_SOURCE_ID,
            top_directory_name=TOP_RAW_DIRECTORY_NAME,
            raise_error_if_missing=False)
        self.assertTrue(this_raw_file_name == RAW_FILE_NAME_MRMS)

    def test_find_raw_file_inexact_time(self):
        """Ensures correct output from find_raw_file_inexact_time."""

        this_raw_file_name = myrorss_and_mrms_io.find_raw_file_inexact_time(
            desired_time_unix_sec=FILE_TIME_UNIX_SEC,
            spc_date_string=FILE_SPC_DATE_STRING,
            field_name=LL_SHEAR_NAME_NEW,
            data_source=radar_utils.MYRORSS_SOURCE_ID,
            top_directory_name=TOP_RAW_DIRECTORY_NAME,
            raise_error_if_missing=False)
        self.assertTrue(this_raw_file_name is None)


if __name__ == '__main__':
    unittest.main()
