"""Unit tests for madis_io.py."""

import copy
import collections
import unittest
import numpy
import pandas
from gewittergefahr.gg_io import myrorss_io
from gewittergefahr.gg_io import raw_wind_io
from gewittergefahr.gg_io import madis_io

TOLERANCE = 1e-6
COLUMN_NAME_ORIG = madis_io.TIME_COLUMN_ORIG
COLUMN_NAME = raw_wind_io.TIME_COLUMN

UNIX_TIME_SEC = 1506127260  # 0041 UTC 23 Sep 2017
YEAR_STRING = '2017'
YEAR_MONTH_STRING = '201709'
MONTH_STRING = '09'
DAY_OF_MONTH_STRING = '23'
TIME_STRING = '20170923_0000'
EXPECTED_PATHLESS_FILE_NAME = '20170923_0000.gz'

SUBDATASET_NAME_LDAD = 'hfmetar'
EXPECTED_FTP_FILE_NAME_LDAD = (
    'archive/2017/09/23/LDAD/hfmetar/netCDF/20170923_0000.gz')
TOP_LOCAL_DIRECTORY_NAME = 'madis_data'
EXPECTED_LOCAL_GZIP_FILE_NAME_LDAD = (
    'madis_data/hfmetar/201709/20170923_0000.gz')

SUBDATASET_NAME_NON_LDAD = 'maritime'
EXPECTED_FTP_FILE_NAME_NON_LDAD = (
    'archive/2017/09/23/point/maritime/netcdf/20170923_0000.gz')
EXPECTED_LOCAL_GZIP_FILE_NAME_NON_LDAD = (
    'madis_data/maritime/201709/20170923_0000.gz')

CHAR_MATRIX = numpy.array([['f', 'o', 'o', 'b', 'a', 'r'],
                           ['f', 'o', 'o', ' ', ' ', ' '],
                           ['m', 'o', 'o', ' ', ' ', ' '],
                           ['h', 'a', 'l', ' ', ' ', ' '],
                           ['p', ' ', 'o', 'o', ' ', 'p']])
EXPECTED_STRING_LIST = ['foobar', 'foo', 'moo', 'hal', 'p oo p']

STATION_IDS_FOR_TABLE = ['CYEG', 'CYYC', 'CYQF', 'CYXH', 'CYQL', 'CYQU', 'CYOD',
                         'CYOJ', 'CYMM']
STATION_NAMES_FOR_TABLE = ['Edmonton', 'Calgary', 'Red Deer', 'Medicine Hat',
                           'Lethbridge', 'Grande Prairie', 'Cold Lake',
                           'High Level', 'Fort McMurray']
LAT_FOR_TABLE_DEG = numpy.array(
    [51.05, 53.55, 52.27, 50.05, 49.7, 55.17, 54.45, 58.52, 56.73])
LNG_FOR_TABLE_DEG = numpy.array(
    [-114.05, -113.47, -113.8, -110.67, -112.82, -118.8, -110.17, -117.13,
     -111.38])
LNG_FOR_TABLE_DEG = myrorss_io.convert_lng_positive_in_west(LNG_FOR_TABLE_DEG)
ELEV_FOR_TABLE_M_ASL = numpy.array(
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

WIND_DICT_NO_ERRORS = {
    raw_wind_io.STATION_ID_COLUMN: STATION_IDS_FOR_TABLE,
    raw_wind_io.STATION_NAME_COLUMN: STATION_NAMES_FOR_TABLE,
    raw_wind_io.LATITUDE_COLUMN: LAT_FOR_TABLE_DEG,
    raw_wind_io.LONGITUDE_COLUMN: LNG_FOR_TABLE_DEG,
    raw_wind_io.ELEVATION_COLUMN: ELEV_FOR_TABLE_M_ASL,
    raw_wind_io.TIME_COLUMN: TIMES_FOR_TABLE_UNIX_SEC,
    raw_wind_io.WIND_SPEED_COLUMN: SPEEDS_FOR_TABLE_M_S01,
    raw_wind_io.WIND_DIR_COLUMN: DIRECTIONS_FOR_TABLE_DEG,
    raw_wind_io.WIND_GUST_SPEED_COLUMN: GUST_SPEEDS_FOR_TABLE_M_S01,
    raw_wind_io.WIND_GUST_DIR_COLUMN: GUST_DIRECTIONS_FOR_TABLE_DEG,
    madis_io.WIND_SPEED_FLAG_COLUMN: QUALITY_FLAGS_FOR_TABLE,
    madis_io.WIND_DIR_FLAG_COLUMN: QUALITY_FLAGS_FOR_TABLE,
    madis_io.WIND_GUST_SPEED_FLAG_COLUMN: QUALITY_FLAGS_FOR_TABLE,
    madis_io.WIND_GUST_DIR_FLAG_COLUMN: QUALITY_FLAGS_FOR_TABLE}

WIND_TABLE_NO_ERRORS = pandas.DataFrame.from_dict(WIND_DICT_NO_ERRORS)

TOO_LOW_VALUE = -5000.
TOO_HIGH_VALUE = 5000.
INVALID_ROWS = [0, 1, 2, 3]

LAT_FOR_TABLE_DEG[0] = TOO_HIGH_VALUE  # Row 0 should be removed.
LNG_FOR_TABLE_DEG[1] = TOO_LOW_VALUE  # Row 1 should be removed.
ELEV_FOR_TABLE_M_ASL[2] = TOO_LOW_VALUE  # Row 2 should be removed.
SPEEDS_FOR_TABLE_M_S01[3] = TOO_LOW_VALUE  # Row 3 should be removed.
GUST_SPEEDS_FOR_TABLE_M_S01[3] = TOO_HIGH_VALUE
SPEEDS_FOR_TABLE_M_S01[4] = TOO_HIGH_VALUE  # Rows 4-8 should be kept.
SPEEDS_FOR_TABLE_M_S01[5] = numpy.nan
GUST_SPEEDS_FOR_TABLE_M_S01[6] = None
DIRECTIONS_FOR_TABLE_DEG[4] = TOO_LOW_VALUE
GUST_DIRECTIONS_FOR_TABLE_DEG[5] = TOO_HIGH_VALUE
DIRECTIONS_FOR_TABLE_DEG[6] = numpy.nan
GUST_DIRECTIONS_FOR_TABLE_DEG[7] = None

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
    raw_wind_io.LATITUDE_COLUMN: LAT_FOR_TABLE_DEG,
    raw_wind_io.LONGITUDE_COLUMN: LNG_FOR_TABLE_DEG,
    raw_wind_io.ELEVATION_COLUMN: ELEV_FOR_TABLE_M_ASL,
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

SPEEDS_FOR_TABLE_M_S01[4] = numpy.nan
GUST_SPEEDS_FOR_TABLE_M_S01[6] = numpy.nan
DIRECTIONS_FOR_TABLE_DEG[4] = raw_wind_io.WIND_DIR_DEFAULT_DEG
GUST_DIRECTIONS_FOR_TABLE_DEG[5] = raw_wind_io.WIND_DIR_DEFAULT_DEG
DIRECTIONS_FOR_TABLE_DEG[6] = raw_wind_io.WIND_DIR_DEFAULT_DEG
GUST_DIRECTIONS_FOR_TABLE_DEG[7] = raw_wind_io.WIND_DIR_DEFAULT_DEG

WIND_TABLE_NO_INVALID_DATA = copy.deepcopy(WIND_TABLE_WITH_ERRORS)

WIND_TABLE_NO_INVALID_DATA[
    raw_wind_io.WIND_SPEED_COLUMN] = SPEEDS_FOR_TABLE_M_S01
WIND_TABLE_NO_INVALID_DATA[
    raw_wind_io.WIND_GUST_SPEED_COLUMN] = GUST_SPEEDS_FOR_TABLE_M_S01
WIND_TABLE_NO_INVALID_DATA[
    raw_wind_io.WIND_DIR_COLUMN] = DIRECTIONS_FOR_TABLE_DEG
WIND_TABLE_NO_INVALID_DATA[
    raw_wind_io.WIND_GUST_DIR_COLUMN] = GUST_DIRECTIONS_FOR_TABLE_DEG
WIND_TABLE_NO_INVALID_DATA.drop(WIND_TABLE_NO_INVALID_DATA.index[INVALID_ROWS],
                                axis=0, inplace=True)

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

    def test_time_unix_sec_to_year_string(self):
        """Ensures correct output from _time_unix_sec_to_year_string."""

        this_year_string = madis_io._time_unix_sec_to_year_string(UNIX_TIME_SEC)
        self.assertTrue(this_year_string == YEAR_STRING)

    def test_time_unix_sec_to_year_month_string(self):
        """Ensures correct output from _time_unix_sec_to_year_month_string."""

        this_year_month_string = madis_io._time_unix_sec_to_year_month_string(
            UNIX_TIME_SEC)
        self.assertTrue(this_year_month_string == YEAR_MONTH_STRING)

    def test_time_unix_sec_to_month_string(self):
        """Ensures correct output from _time_unix_sec_to_month_string."""

        this_month_string = madis_io._time_unix_sec_to_month_string(
            UNIX_TIME_SEC)
        self.assertTrue(this_month_string == MONTH_STRING)

    def test_time_unix_sec_to_day_of_month_string(self):
        """Ensures correct output from _time_unix_sec_to_day_of_month_string."""

        this_day_of_month_string = (
            madis_io._time_unix_sec_to_day_of_month_string(UNIX_TIME_SEC))
        self.assertTrue(this_day_of_month_string == DAY_OF_MONTH_STRING)

    def test_time_unix_sec_to_string(self):
        """Ensures correct output from _time_unix_sec_to_string."""

        this_time_string = madis_io._time_unix_sec_to_string(UNIX_TIME_SEC)
        self.assertTrue(this_time_string == TIME_STRING)

    def test_get_ftp_file_name_ldad(self):
        """Ensures correct output from _get_ftp_file_name.
        
        In this case, subdataset is HFMETAR, which is part of the LDAD (Local
        Data Acquisition and Dissemination) system.
        """

        this_ftp_file_name = madis_io._get_ftp_file_name(
            UNIX_TIME_SEC, SUBDATASET_NAME_LDAD)
        self.assertTrue(this_ftp_file_name == EXPECTED_FTP_FILE_NAME_LDAD)

    def test_get_ftp_file_name_non_ldad(self):
        """Ensures correct output from _get_ftp_file_name.

        In this case, subdataset is maritime, which is *not* part of LDAD.
        """

        this_ftp_file_name = madis_io._get_ftp_file_name(
            UNIX_TIME_SEC, SUBDATASET_NAME_NON_LDAD)
        self.assertTrue(this_ftp_file_name == EXPECTED_FTP_FILE_NAME_NON_LDAD)

    def test_find_local_raw_file_ldad(self):
        """Ensures correct output from find_local_raw_file.

        In this case, subdataset is HFMETAR, which is part of LDAD.
        """

        this_file_name = madis_io.find_local_raw_file(
            unix_time_sec=UNIX_TIME_SEC, subdataset_name=SUBDATASET_NAME_LDAD,
            file_extension=madis_io.GZIP_FILE_EXTENSION,
            top_local_directory_name=TOP_LOCAL_DIRECTORY_NAME,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == EXPECTED_LOCAL_GZIP_FILE_NAME_LDAD)

    def test_find_local_raw_file_non_ldad(self):
        """Ensures correct output from find_local_raw_file.

        In this case, subdataset is maritime, which is *not* part of LDAD.
        """

        this_file_name = madis_io.find_local_raw_file(
            unix_time_sec=UNIX_TIME_SEC,
            subdataset_name=SUBDATASET_NAME_NON_LDAD,
            file_extension=madis_io.GZIP_FILE_EXTENSION,
            top_local_directory_name=TOP_LOCAL_DIRECTORY_NAME,
            raise_error_if_missing=False)

        self.assertTrue(
            this_file_name == EXPECTED_LOCAL_GZIP_FILE_NAME_NON_LDAD)

    def test_convert_column_name(self):
        """Ensures correct output from _convert_column_name."""

        this_column_name = madis_io._convert_column_name(COLUMN_NAME_ORIG)
        self.assertTrue(this_column_name == COLUMN_NAME)

    def test_char_matrix_to_string_list(self):
        """Ensures correct output from _char_matrix_to_string_list."""

        string_list = madis_io._char_matrix_to_string_list(CHAR_MATRIX)
        self.assertTrue(collections.Counter(string_list) == collections.Counter(
            EXPECTED_STRING_LIST))

    def test_remove_invalid_data_none_invalid(self):
        """Ensures correctness of _remove_invalid_data with no invalid data."""

        this_wind_table = madis_io._remove_invalid_data(WIND_TABLE_NO_ERRORS)
        self.assertTrue(this_wind_table.equals(WIND_TABLE_NO_ERRORS))

    def test_remove_invalid_data_some_invalid(self):
        """Correctness of _remove_invalid_data with some invalid data."""

        this_table_with_errors = copy.deepcopy(WIND_TABLE_WITH_ERRORS)
        this_wind_table = madis_io._remove_invalid_data(this_table_with_errors)
        self.assertTrue(this_wind_table.equals(WIND_TABLE_NO_INVALID_DATA))

    def test_remove_low_quality_data_no_low_quality(self):
        """Correctness of _remove_low_quality_data with no low-quality data."""

        this_wind_table = madis_io._remove_low_quality_data(
            WIND_TABLE_NO_ERRORS)
        self.assertTrue(this_wind_table.equals(WIND_TABLE_NO_ERRORS))

    def test_remove_low_quality_data_some_low_quality(self):
        """Correctness of _remove_low_quality_data, some low-quality data."""

        this_table_with_errors = copy.deepcopy(WIND_TABLE_WITH_ERRORS)
        this_wind_table = madis_io._remove_low_quality_data(
            this_table_with_errors)

        self.assertTrue(this_wind_table.equals(WIND_TABLE_NO_LOW_QUALITY_DATA))

    def test_get_pathless_raw_file_name(self):
        """Ensures correct output from _get_pathless_raw_file_name."""

        this_pathless_file_name = madis_io._get_pathless_raw_file_name(
            UNIX_TIME_SEC, madis_io.GZIP_FILE_EXTENSION)
        self.assertTrue(this_pathless_file_name == EXPECTED_PATHLESS_FILE_NAME)


if __name__ == '__main__':
    unittest.main()
