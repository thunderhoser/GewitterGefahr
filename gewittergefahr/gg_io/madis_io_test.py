"""Unit tests for madis_io.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_io import raw_wind_io
from gewittergefahr.gg_io import madis_io
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion

COLUMN_NAME = raw_wind_io.TIME_COLUMN
COLUMN_NAME_ORIG = madis_io.TIME_COLUMN_ORIG

CHAR_MATRIX = numpy.array([['f', 'o', 'o', 'b', 'a', 'r'],
                           ['f', 'o', 'o', ' ', ' ', ' '],
                           ['m', 'o', 'o', ' ', ' ', ' '],
                           ['h', 'a', 'l', ' ', ' ', ' '],
                           ['p', ' ', 'o', 'o', ' ', 'p']])
STRING_LIST = ['foobar', 'foo', 'moo', 'hal', 'p oo p']

UNIX_TIME_SEC = 1506127260  # 0041 UTC 23 Sep 2017
PATHLESS_ZIPPED_FILE_NAME = '20170923_0000.gz'
PATHLESS_UNZIPPED_FILE_NAME = '20170923_0000.netcdf'

SUBDATASET_NAME_LDAD = 'hfmetar'
FTP_FILE_NAME_LDAD = 'archive/2017/09/23/LDAD/hfmetar/netCDF/20170923_0000.gz'
TOP_LOCAL_DIRECTORY_NAME = 'madis_data'
ZIPPED_FILE_NAME_LDAD = 'madis_data/hfmetar/201709/20170923_0000.gz'
UNZIPPED_FILE_NAME_LDAD = 'madis_data/hfmetar/201709/20170923_0000.netcdf'

SUBDATASET_NAME_NON_LDAD = 'maritime'
FTP_FILE_NAME_NON_LDAD = (
    'archive/2017/09/23/point/maritime/netcdf/20170923_0000.gz')
ZIPPED_FILE_NAME_NON_LDAD = 'madis_data/maritime/201709/20170923_0000.gz'
UNZIPPED_FILE_NAME_NON_LDAD = 'madis_data/maritime/201709/20170923_0000.netcdf'

STATION_IDS_FOR_TABLE = ['CYEG', 'CYYC', 'CYQF', 'CYXH', 'CYQL', 'CYQU', 'CYOD',
                         'CYOJ', 'CYMM']
STATION_NAMES_FOR_TABLE = ['Edmonton', 'Calgary', 'Red Deer', 'Medicine Hat',
                           'Lethbridge', 'Grande Prairie', 'Cold Lake',
                           'High Level', 'Fort McMurray']
LATITUDES_FOR_TABLE_DEG = numpy.array(
    [51.05, 53.55, 52.27, 50.05, 49.7, 55.17, 54.45, 58.52, 56.73])
LONGITUDES_FOR_TABLE_DEG = numpy.array(
    [-114.05, -113.47, -113.8, -110.67, -112.82, -118.8, -110.17, -117.13,
     -111.38])
LONGITUDES_FOR_TABLE_DEG = (
    lng_conversion.convert_lng_positive_in_west(LONGITUDES_FOR_TABLE_DEG))
ELEVATIONS_FOR_TABLE_M_ASL = numpy.array(
    [723., 1084., 905., 717., 929., 669., 541., 338., 369.])
TIMES_FOR_TABLE_UNIX_SEC = numpy.linspace(1505360794, 1505360794,
                                          num=len(STATION_IDS_FOR_TABLE),
                                          dtype=int)
SPEEDS_FOR_TABLE_M_S01 = numpy.array([0., 5., 10., 7., 13., 6., 2., 8., 3.])
DIRECTIONS_FOR_TABLE_DEG = numpy.array(
    [0., 20., 100., 150., 330., 225., 200., 270., 45.])
GUST_SPEEDS_FOR_TABLE_M_S01 = numpy.array(
    [2.5, 7.5, 13., 5., 9., 3.5, 4., 5.5, 6.])
GUST_DIRECTIONS_FOR_TABLE_DEG = numpy.array(
    [350., 30., 85., 165., 345., 250., 185., 280., 30.])
QUALITY_FLAGS_FOR_TABLE = [madis_io.DEFAULT_QUALITY_FLAG] * len(
    STATION_IDS_FOR_TABLE)

WIND_DICT_PERFECT_DATA = {
    raw_wind_io.STATION_ID_COLUMN: STATION_IDS_FOR_TABLE,
    raw_wind_io.STATION_NAME_COLUMN: STATION_NAMES_FOR_TABLE,
    raw_wind_io.LATITUDE_COLUMN: LATITUDES_FOR_TABLE_DEG,
    raw_wind_io.LONGITUDE_COLUMN: LONGITUDES_FOR_TABLE_DEG,
    raw_wind_io.ELEVATION_COLUMN: ELEVATIONS_FOR_TABLE_M_ASL,
    raw_wind_io.TIME_COLUMN: TIMES_FOR_TABLE_UNIX_SEC,
    raw_wind_io.WIND_SPEED_COLUMN: SPEEDS_FOR_TABLE_M_S01,
    raw_wind_io.WIND_DIR_COLUMN: DIRECTIONS_FOR_TABLE_DEG,
    raw_wind_io.WIND_GUST_SPEED_COLUMN: GUST_SPEEDS_FOR_TABLE_M_S01,
    raw_wind_io.WIND_GUST_DIR_COLUMN: GUST_DIRECTIONS_FOR_TABLE_DEG,
    madis_io.WIND_SPEED_FLAG_COLUMN: QUALITY_FLAGS_FOR_TABLE,
    madis_io.WIND_DIR_FLAG_COLUMN: QUALITY_FLAGS_FOR_TABLE,
    madis_io.WIND_GUST_SPEED_FLAG_COLUMN: QUALITY_FLAGS_FOR_TABLE,
    madis_io.WIND_GUST_DIR_FLAG_COLUMN: QUALITY_FLAGS_FOR_TABLE}

WIND_TABLE_PERFECT_DATA = pandas.DataFrame.from_dict(WIND_DICT_PERFECT_DATA)

# Add gross errors to rows 0-3, small errors to rows 4-7.
TOO_LOW_VALUE = -5000.
TOO_HIGH_VALUE = 5000.
INVALID_ROWS = [0, 1, 2, 3]

LATITUDES_FOR_TABLE_DEG[0] = TOO_HIGH_VALUE
LONGITUDES_FOR_TABLE_DEG[1] = TOO_LOW_VALUE
ELEVATIONS_FOR_TABLE_M_ASL[2] = TOO_LOW_VALUE
SPEEDS_FOR_TABLE_M_S01[3] = TOO_LOW_VALUE
GUST_SPEEDS_FOR_TABLE_M_S01[3] = TOO_HIGH_VALUE
SPEEDS_FOR_TABLE_M_S01[4] = TOO_HIGH_VALUE
SPEEDS_FOR_TABLE_M_S01[5] = numpy.nan
GUST_SPEEDS_FOR_TABLE_M_S01[6] = None
DIRECTIONS_FOR_TABLE_DEG[4] = TOO_LOW_VALUE
GUST_DIRECTIONS_FOR_TABLE_DEG[5] = TOO_HIGH_VALUE
DIRECTIONS_FOR_TABLE_DEG[6] = numpy.nan
GUST_DIRECTIONS_FOR_TABLE_DEG[7] = None

# Add low-quality flags to some measurements.
LOW_QUALITY_ROWS_SUSTAINED = numpy.array([0, 2, 3, 5], dtype=int)
LOW_QUALITY_ROWS_GUST = numpy.array([0, 2, 3, 8], dtype=int)
LOW_QUALITY_ROWS_SUSTAINED_AND_GUST = numpy.array([0, 2, 3], dtype=int)

QUALITY_FLAGS_SUSTAINED = copy.deepcopy(QUALITY_FLAGS_FOR_TABLE)
for this_row in LOW_QUALITY_ROWS_SUSTAINED:
    QUALITY_FLAGS_SUSTAINED[this_row] = madis_io.LOW_QUALITY_FLAGS[0]

QUALITY_FLAGS_GUST = copy.deepcopy(QUALITY_FLAGS_FOR_TABLE)
for this_row in LOW_QUALITY_ROWS_GUST:
    QUALITY_FLAGS_GUST[this_row] = madis_io.LOW_QUALITY_FLAGS[0]

WIND_DICT_WITH_ERRORS = {
    raw_wind_io.STATION_ID_COLUMN: STATION_IDS_FOR_TABLE,
    raw_wind_io.STATION_NAME_COLUMN: STATION_NAMES_FOR_TABLE,
    raw_wind_io.LATITUDE_COLUMN: LATITUDES_FOR_TABLE_DEG,
    raw_wind_io.LONGITUDE_COLUMN: LONGITUDES_FOR_TABLE_DEG,
    raw_wind_io.ELEVATION_COLUMN: ELEVATIONS_FOR_TABLE_M_ASL,
    raw_wind_io.TIME_COLUMN: TIMES_FOR_TABLE_UNIX_SEC,
    raw_wind_io.WIND_SPEED_COLUMN: SPEEDS_FOR_TABLE_M_S01,
    raw_wind_io.WIND_DIR_COLUMN: DIRECTIONS_FOR_TABLE_DEG,
    raw_wind_io.WIND_GUST_SPEED_COLUMN: GUST_SPEEDS_FOR_TABLE_M_S01,
    raw_wind_io.WIND_GUST_DIR_COLUMN: GUST_DIRECTIONS_FOR_TABLE_DEG,
    madis_io.WIND_SPEED_FLAG_COLUMN: QUALITY_FLAGS_SUSTAINED,
    madis_io.WIND_DIR_FLAG_COLUMN: QUALITY_FLAGS_SUSTAINED,
    madis_io.WIND_GUST_SPEED_FLAG_COLUMN: QUALITY_FLAGS_GUST,
    madis_io.WIND_GUST_DIR_FLAG_COLUMN: QUALITY_FLAGS_GUST}

WIND_TABLE_WITH_ERRORS = pandas.DataFrame.from_dict(WIND_DICT_WITH_ERRORS)

# Create table with no errors (but some low-quality data).
SPEEDS_FOR_TABLE_M_S01[4] = numpy.nan
GUST_SPEEDS_FOR_TABLE_M_S01[6] = numpy.nan
DIRECTIONS_FOR_TABLE_DEG[4] = raw_wind_io.WIND_DIR_DEFAULT_DEG
GUST_DIRECTIONS_FOR_TABLE_DEG[5] = raw_wind_io.WIND_DIR_DEFAULT_DEG
DIRECTIONS_FOR_TABLE_DEG[6] = raw_wind_io.WIND_DIR_DEFAULT_DEG
GUST_DIRECTIONS_FOR_TABLE_DEG[7] = raw_wind_io.WIND_DIR_DEFAULT_DEG

WIND_TABLE_NO_ERRORS = copy.deepcopy(WIND_TABLE_WITH_ERRORS)
WIND_TABLE_NO_ERRORS[raw_wind_io.WIND_SPEED_COLUMN] = SPEEDS_FOR_TABLE_M_S01
WIND_TABLE_NO_ERRORS[
    raw_wind_io.WIND_GUST_SPEED_COLUMN] = GUST_SPEEDS_FOR_TABLE_M_S01
WIND_TABLE_NO_ERRORS[raw_wind_io.WIND_DIR_COLUMN] = DIRECTIONS_FOR_TABLE_DEG
WIND_TABLE_NO_ERRORS[
    raw_wind_io.WIND_GUST_DIR_COLUMN] = GUST_DIRECTIONS_FOR_TABLE_DEG
WIND_TABLE_NO_ERRORS.drop(WIND_TABLE_NO_ERRORS.index[INVALID_ROWS], axis=0,
                          inplace=True)

# Create table with no low-quality data (but some errors).
WIND_TABLE_NO_LOW_QUALITY_DATA = copy.deepcopy(WIND_TABLE_WITH_ERRORS)
WIND_TABLE_NO_LOW_QUALITY_DATA[raw_wind_io.WIND_SPEED_COLUMN].values[
    LOW_QUALITY_ROWS_SUSTAINED] = numpy.nan
WIND_TABLE_NO_LOW_QUALITY_DATA[raw_wind_io.WIND_DIR_COLUMN].values[
    LOW_QUALITY_ROWS_SUSTAINED] = raw_wind_io.WIND_DIR_DEFAULT_DEG
WIND_TABLE_NO_LOW_QUALITY_DATA[raw_wind_io.WIND_GUST_SPEED_COLUMN].values[
    LOW_QUALITY_ROWS_GUST] = numpy.nan
WIND_TABLE_NO_LOW_QUALITY_DATA[raw_wind_io.WIND_GUST_DIR_COLUMN].values[
    LOW_QUALITY_ROWS_GUST] = raw_wind_io.WIND_DIR_DEFAULT_DEG
WIND_TABLE_NO_LOW_QUALITY_DATA.drop(
    WIND_TABLE_NO_LOW_QUALITY_DATA.index[LOW_QUALITY_ROWS_SUSTAINED_AND_GUST],
    axis=0, inplace=True)

FLAG_COLUMNS = [madis_io.WIND_SPEED_FLAG_COLUMN, madis_io.WIND_DIR_FLAG_COLUMN,
                madis_io.WIND_GUST_SPEED_FLAG_COLUMN,
                madis_io.WIND_GUST_DIR_FLAG_COLUMN]
WIND_TABLE_NO_LOW_QUALITY_DATA.drop(FLAG_COLUMNS, axis=1, inplace=True)


class MadisIoTests(unittest.TestCase):
    """Each method is a unit test for madis_io.py."""

    def test_column_name_orig_to_new(self):
        """Ensures correct output from _column_name_orig_to_new."""

        this_column_name = madis_io._column_name_orig_to_new(COLUMN_NAME_ORIG)
        self.assertTrue(this_column_name == COLUMN_NAME)

    def test_char_matrix_to_string_list(self):
        """Ensures correct output from _char_matrix_to_string_list."""

        string_list = madis_io._char_matrix_to_string_list(CHAR_MATRIX)
        self.assertTrue(len(string_list) == len(STRING_LIST))
        for i in range(len(string_list)):
            self.assertTrue(string_list[i] == STRING_LIST[i])

    def test_get_ftp_file_name_ldad(self):
        """Ensures correct output from _get_ftp_file_name.

        In this case, subdataset is HFMETAR, which is part of the LDAD (Local
        Data Acquisition and Dissemination) system.
        """

        this_ftp_file_name = madis_io._get_ftp_file_name(
            UNIX_TIME_SEC, SUBDATASET_NAME_LDAD)
        self.assertTrue(this_ftp_file_name == FTP_FILE_NAME_LDAD)

    def test_get_ftp_file_name_non_ldad(self):
        """Ensures correct output from _get_ftp_file_name.

        In this case, subdataset is maritime, which is *not* part of LDAD.
        """

        this_ftp_file_name = madis_io._get_ftp_file_name(
            UNIX_TIME_SEC, SUBDATASET_NAME_NON_LDAD)
        self.assertTrue(this_ftp_file_name == FTP_FILE_NAME_NON_LDAD)

    def test_remove_invalid_data_none_invalid(self):
        """Ensures correctness of _remove_invalid_data with no invalid data."""

        this_wind_table = madis_io._remove_invalid_data(WIND_TABLE_PERFECT_DATA)
        self.assertTrue(this_wind_table.equals(WIND_TABLE_PERFECT_DATA))

    def test_remove_invalid_data_some_invalid(self):
        """Correctness of _remove_invalid_data with some invalid data."""

        this_table_with_errors = copy.deepcopy(WIND_TABLE_WITH_ERRORS)
        this_wind_table = madis_io._remove_invalid_data(this_table_with_errors)
        self.assertTrue(this_wind_table.equals(WIND_TABLE_NO_ERRORS))

    def test_remove_low_quality_data_no_low_quality(self):
        """Correctness of _remove_low_quality_data with no low-quality data."""

        this_wind_table = madis_io._remove_low_quality_data(
            WIND_TABLE_PERFECT_DATA)
        self.assertTrue(this_wind_table.equals(WIND_TABLE_PERFECT_DATA))

    def test_remove_low_quality_data_some_low_quality(self):
        """Correctness of _remove_low_quality_data, some low-quality data."""

        this_table_with_errors = copy.deepcopy(WIND_TABLE_WITH_ERRORS)
        this_wind_table = madis_io._remove_low_quality_data(
            this_table_with_errors)

        self.assertTrue(this_wind_table.equals(WIND_TABLE_NO_LOW_QUALITY_DATA))

    def test_get_pathless_raw_file_name_zipped(self):
        """Ensures correct output from _get_pathless_raw_file_name.

        In this case, looking for zipped file.
        """

        this_pathless_file_name = madis_io._get_pathless_raw_file_name(
            UNIX_TIME_SEC, zipped=True)
        self.assertTrue(this_pathless_file_name == PATHLESS_ZIPPED_FILE_NAME)

    def test_get_pathless_raw_file_name_unzipped(self):
        """Ensures correct output from _get_pathless_raw_file_name.

        In this case, looking for unzipped file.
        """

        this_pathless_file_name = madis_io._get_pathless_raw_file_name(
            UNIX_TIME_SEC, zipped=False)
        self.assertTrue(this_pathless_file_name == PATHLESS_UNZIPPED_FILE_NAME)

    def test_find_local_raw_file_ldad_zipped(self):
        """Ensures correct output from find_local_raw_file.

        In this case, looking for zipped file from subdataset HFMETAR, which is
        part of LDAD.
        """

        this_file_name = madis_io.find_local_raw_file(
            unix_time_sec=UNIX_TIME_SEC, subdataset_name=SUBDATASET_NAME_LDAD,
            top_local_directory_name=TOP_LOCAL_DIRECTORY_NAME, zipped=True,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == ZIPPED_FILE_NAME_LDAD)

    def test_find_local_raw_file_ldad_unzipped(self):
        """Ensures correct output from find_local_raw_file.

        In this case, looking for unzipped file from subdataset HFMETAR, which
        is part of LDAD.
        """

        this_file_name = madis_io.find_local_raw_file(
            unix_time_sec=UNIX_TIME_SEC, subdataset_name=SUBDATASET_NAME_LDAD,
            top_local_directory_name=TOP_LOCAL_DIRECTORY_NAME, zipped=False,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == UNZIPPED_FILE_NAME_LDAD)

    def test_find_local_raw_file_non_ldad_zipped(self):
        """Ensures correct output from find_local_raw_file.

        In this case, looking for zipped file from subdataset maritime, which is
        not part of LDAD.
        """

        this_file_name = madis_io.find_local_raw_file(
            unix_time_sec=UNIX_TIME_SEC,
            subdataset_name=SUBDATASET_NAME_NON_LDAD,
            top_local_directory_name=TOP_LOCAL_DIRECTORY_NAME, zipped=True,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == ZIPPED_FILE_NAME_NON_LDAD)

    def test_find_local_raw_file_non_ldad_unzipped(self):
        """Ensures correct output from find_local_raw_file.

        In this case, looking for unzipped file from subdataset maritime, which
        is not part of LDAD.
        """

        this_file_name = madis_io.find_local_raw_file(
            unix_time_sec=UNIX_TIME_SEC,
            subdataset_name=SUBDATASET_NAME_NON_LDAD,
            top_local_directory_name=TOP_LOCAL_DIRECTORY_NAME, zipped=False,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == UNZIPPED_FILE_NAME_NON_LDAD)


if __name__ == '__main__':
    unittest.main()
