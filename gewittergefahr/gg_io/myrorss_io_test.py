"""Unit tests for myrorss_io.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_io import myrorss_io

TOLERANCE = 1e-6

UNIX_TIME_SEC_1200UTC = 1506168000
UNIX_TIME_SEC_0000UTC = 1506211200
UNIX_TIME_SEC_115959UTC = 1506254399
SPC_DATE_STRING = '20170923'

UNIX_TIME_SEC = 1506202714
TIME_STRING = '20170923-213834'

VARIABLE_NAME_0METRES = myrorss_io.MID_LEVEL_SHEAR_NAME
VARIABLE_NAME_250METRES = myrorss_io.SHI_NAME
VARIABLE_NAME_MANY_HEIGHTS = myrorss_io.REFL_NAME

VARIABLE_NAMES_FOR_DICT = [myrorss_io.LOW_LEVEL_SHEAR_NAME,
                           myrorss_io.REFL_M10CELSIUS_NAME,
                           myrorss_io.REFL_NAME]
REFL_HEIGHTS_M_AGL = numpy.array([0., 1000., 2000., 3000., 5000., 10000.])
EXPECTED_VARIABLE_DICT = {myrorss_io.LOW_LEVEL_SHEAR_NAME: numpy.array([0.]),
                          myrorss_io.REFL_M10CELSIUS_NAME: numpy.array([250.]),
                          myrorss_io.REFL_NAME: REFL_HEIGHTS_M_AGL}

TIME_FOR_FILE_UNIX_SEC = 1286705410  # 101010 UTC 10 Oct 2010
SPC_DATE_FOR_FILE_UNIX_SEC = 1286705410  # any time from 1200 UTC 9 Oct - 10 Oct
VARIABLE_NAME_FOR_FILE = myrorss_io.REFL_M20CELSIUS_NAME
HEIGHT_FOR_FILE_M_AGL = 250.
EXPECTED_DIR_NAME_IN_TAR_FILE = myrorss_io.REFL_M20CELSIUS_NAME_ORIG + '/00.25'
EXPECTED_PATHLESS_GZIP_FILE_NAME = '20101010-101010.netcdf.gz'
EXPECTED_PATHLESS_NETCDF_FILE_NAME = '20101010-101010.netcdf'

TOP_RAW_DIRECTORY_NAME = 'myrorss'
EXPECTED_GZIP_FILE_NAME = (
    'myrorss/20101009/' + myrorss_io.REFL_M20CELSIUS_NAME_ORIG +
    '/00.25/20101010-101010.netcdf.gz')
EXPECTED_NETCDF_FILE_NAME = (
    'myrorss/20101009/' + myrorss_io.REFL_M20CELSIUS_NAME_ORIG +
    '/00.25/20101010-101010.netcdf')

VARIABLE_NAME_NEW = myrorss_io.REFL_COLUMN_MAX_NAME
VARIABLE_NAME_ORIG = myrorss_io.REFL_COLUMN_MAX_NAME_ORIG

GRID_ROWS = numpy.linspace(0, 10, num=11, dtype=int)
GRID_COLUMNS = numpy.linspace(0, 10, num=11, dtype=int)
GRID_CELL_COUNTS = numpy.linspace(0, 10, num=11, dtype=int)
RADAR_VALUES = numpy.linspace(0, 10, num=11)

SENTINEL_VALUES = numpy.array([-99000., -99001.])
SENTINEL_INDICES = numpy.array([7, 9], dtype=int)
RADAR_VALUES[SENTINEL_INDICES[0]] = SENTINEL_VALUES[0]
RADAR_VALUES[SENTINEL_INDICES[1]] = SENTINEL_VALUES[1]

SPARSE_GRID_DICT_WITH_SENTINELS = {myrorss_io.GRID_ROW_COLUMN: GRID_ROWS,
                                   myrorss_io.GRID_COLUMN_COLUMN: GRID_COLUMNS,
                                   myrorss_io.NUM_GRID_CELL_COLUMN:
                                       GRID_CELL_COUNTS,
                                   VARIABLE_NAME_NEW: RADAR_VALUES}
SPARSE_GRID_TABLE_WITH_SENTINELS = pandas.DataFrame.from_dict(
    SPARSE_GRID_DICT_WITH_SENTINELS)

EXPECTED_SPARSE_GRID_TABLE_NO_SENTINELS = copy.deepcopy(
    SPARSE_GRID_TABLE_WITH_SENTINELS)
EXPECTED_SPARSE_GRID_TABLE_NO_SENTINELS.drop(
    EXPECTED_SPARSE_GRID_TABLE_NO_SENTINELS.index[SENTINEL_INDICES], axis=0,
    inplace=True)

LONGITUDES_DEG = numpy.array(
    [0., 45., 90., 135., 180., 210., -120., 270., -60., 330.])
EXPECTED_LNG_NEGATIVE_IN_WEST_DEG = numpy.array(
    [0., 45., 90., 135., 180., -150., -120., -90., -60., -30.])
EXPECTED_LNG_POSITIVE_IN_WEST_DEG = numpy.array(
    [0., 45., 90., 135., 180., 210., 240., 270., 300., 330.])

NW_GRID_POINT_LAT_DEG = 55.
NW_GRID_POINT_LNG_DEG = 230.
LAT_SPACING_DEG = 0.01
LNG_SPACING_DEG = 0.01
EXPECTED_GRID_LAT_DEG = numpy.array(
    [55., 54.99, 54.98, 54.97, 54.96, 54.95, 54.94, 54.93, 54.92, 54.91, 54.9])
EXPECTED_GRID_LNG_DEG = numpy.array(
    [230., 230.01, 230.02, 230.03, 230.04, 230.05, 230.06, 230.07, 230.08,
     230.09, 230.1])

NUM_LAT_IN_GRID = 3501
NUM_LNG_IN_GRID = 7001
CENTER_LAT_IN_GRID_DEG = 37.5
CENTER_LNG_IN_GRID_DEG = 265.


class MyrorssIoTests(unittest.TestCase):
    """Each method is a unit test for myrorss_io.py."""

    def test_time_unix_sec_to_spc_date_1200utc(self):
        """Ensures correct output from _time_unix_sec_to_spc_date.

        In this case the time is 1200 UTC 23 Sep 2017 (beginning of SPC date
        "20170923").
        """

        this_spc_date_string = myrorss_io.time_unix_sec_to_spc_date(
            UNIX_TIME_SEC_1200UTC)
        self.assertTrue(this_spc_date_string == SPC_DATE_STRING)

    def test_time_unix_sec_to_spc_date_0000utc(self):
        """Ensures correct output from _time_unix_sec_to_spc_date.

        In this case the time is 0000 UTC 24 Sep 2017 (middle of SPC date
        "20170923").
        """

        this_spc_date_string = myrorss_io.time_unix_sec_to_spc_date(
            UNIX_TIME_SEC_0000UTC)
        self.assertTrue(this_spc_date_string == SPC_DATE_STRING)

    def test_time_unix_sec_to_spc_date_115959utc(self):
        """Ensures correct output from _time_unix_sec_to_spc_date.

        In this case the time is 115959 UTC 24 Sep 2017 (end of SPC date
        "20170923").
        """

        this_spc_date_string = myrorss_io.time_unix_sec_to_spc_date(
            UNIX_TIME_SEC_115959UTC)
        self.assertTrue(this_spc_date_string == SPC_DATE_STRING)

    def test_time_unix_sec_to_string(self):
        """Ensures correct output from _time_unix_sec_to_string."""

        this_time_string = myrorss_io._time_unix_sec_to_string(UNIX_TIME_SEC)
        self.assertTrue(this_time_string == TIME_STRING)

    def test_variable_to_valid_heights_0metres(self):
        """Ensures correct output from _variable_to_valid_heights.

        In this case the only valid height is 0 metres.
        """

        these_heights_m_agl = myrorss_io._variable_to_valid_heights(
            VARIABLE_NAME_0METRES)
        self.assertTrue(numpy.allclose(these_heights_m_agl, numpy.array([0.]),
                                       atol=TOLERANCE))

    def test_variable_to_valid_heights_250metres(self):
        """Ensures correct output from _variable_to_valid_heights.

        In this case the only valid height is 250 metres.
        """

        these_heights_m_agl = myrorss_io._variable_to_valid_heights(
            VARIABLE_NAME_250METRES)
        self.assertTrue(
            numpy.allclose(these_heights_m_agl, numpy.array([250.]),
                           atol=TOLERANCE))

    def test_variable_to_valid_heights_many(self):
        """Ensures correct output from _variable_to_valid_heights.

        In this case there are many valid heights.
        """

        these_heights_m_agl = myrorss_io._variable_to_valid_heights(
            VARIABLE_NAME_MANY_HEIGHTS)
        self.assertTrue(len(these_heights_m_agl) > 1)

    def test_var_height_arrays_to_dict(self):
        """Ensures correct output from _var_height_arrays_to_dict."""

        this_variable_dict = myrorss_io._var_height_arrays_to_dict(
            VARIABLE_NAMES_FOR_DICT, REFL_HEIGHTS_M_AGL)
        self.assertTrue(
            this_variable_dict.keys() == EXPECTED_VARIABLE_DICT.keys())

        for j in range(len(VARIABLE_NAMES_FOR_DICT)):
            this_key = VARIABLE_NAMES_FOR_DICT[j]
            self.assertTrue(numpy.allclose(this_variable_dict[this_key],
                                           EXPECTED_VARIABLE_DICT[this_key],
                                           atol=TOLERANCE))

    def test_get_directory_in_tar_file(self):
        """Ensures correct output from _get_directory_in_tar_file."""

        this_directory_name = myrorss_io._get_directory_in_tar_file(
            VARIABLE_NAME_FOR_FILE, HEIGHT_FOR_FILE_M_AGL)
        self.assertTrue(this_directory_name == EXPECTED_DIR_NAME_IN_TAR_FILE)

    def test_get_pathless_gzip_file_name(self):
        """Ensures correct output from _get_pathless_raw_file_name.

        In this case the expected file is a gzip archive.
        """

        this_pathless_file_name = myrorss_io._get_pathless_raw_file_name(
            TIME_FOR_FILE_UNIX_SEC, zipped=True)
        self.assertTrue(
            this_pathless_file_name == EXPECTED_PATHLESS_GZIP_FILE_NAME)

    def test_get_pathless_netcdf_file_name(self):
        """Ensures correct output from _get_pathless_raw_file_name.

        In this case the expected file is a NetCDF.
        """

        this_pathless_file_name = myrorss_io._get_pathless_raw_file_name(
            TIME_FOR_FILE_UNIX_SEC, zipped=False)
        self.assertTrue(
            this_pathless_file_name == EXPECTED_PATHLESS_NETCDF_FILE_NAME)

    def test_var_name_orig_to_new(self):
        """Ensures correct output from _var_name_orig_to_new."""

        this_var_name_new = myrorss_io._var_name_orig_to_new(VARIABLE_NAME_ORIG)
        self.assertTrue(this_var_name_new == VARIABLE_NAME_NEW)

    def test_var_name_new_to_orig(self):
        """Ensures correct output from _var_name_new_to_orig."""

        this_var_name_orig = myrorss_io._var_name_new_to_orig(VARIABLE_NAME_NEW)
        self.assertTrue(this_var_name_orig == VARIABLE_NAME_ORIG)

    def test_find_local_gzip_file(self):
        """Ensures correct output from find_local_gzip_file.

        In this case the expected file is a gzip archive.
        """

        this_file_name = myrorss_io.find_local_raw_file(
            unix_time_sec=TIME_FOR_FILE_UNIX_SEC,
            spc_date_unix_sec=SPC_DATE_FOR_FILE_UNIX_SEC,
            variable_name=VARIABLE_NAME_FOR_FILE,
            height_m_agl=HEIGHT_FOR_FILE_M_AGL,
            top_directory_name=TOP_RAW_DIRECTORY_NAME, zipped=True,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == EXPECTED_GZIP_FILE_NAME)

    def test_find_local_netcdf_file(self):
        """Ensures correct output from find_local_gzip_file.

        In this case the expected file is a NetCDF.
        """

        this_file_name = myrorss_io.find_local_raw_file(
            unix_time_sec=TIME_FOR_FILE_UNIX_SEC,
            spc_date_unix_sec=SPC_DATE_FOR_FILE_UNIX_SEC,
            variable_name=VARIABLE_NAME_FOR_FILE,
            height_m_agl=HEIGHT_FOR_FILE_M_AGL,
            top_directory_name=TOP_RAW_DIRECTORY_NAME, zipped=False,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == EXPECTED_NETCDF_FILE_NAME)

    def test_remove_sentinels(self):
        """Ensures correct output from _remove_sentinels."""

        sparse_grid_table_no_sentinels = myrorss_io._remove_sentinels(
            SPARSE_GRID_TABLE_WITH_SENTINELS, VARIABLE_NAME_NEW,
            SENTINEL_VALUES)
        self.assertTrue(sparse_grid_table_no_sentinels.equals(
            EXPECTED_SPARSE_GRID_TABLE_NO_SENTINELS))

    def test_convert_lng_negative_in_west(self):
        """Ensures correct output from convert_lng_negative_in_west."""

        lng_negative_in_west_deg = myrorss_io.convert_lng_negative_in_west(
            LONGITUDES_DEG)
        self.assertTrue(numpy.allclose(lng_negative_in_west_deg,
                                       EXPECTED_LNG_NEGATIVE_IN_WEST_DEG,
                                       atol=TOLERANCE))

    def test_convert_lng_positive_in_west(self):
        """Ensures correct output from convert_lng_positive_in_west."""

        lng_positive_in_west_deg = myrorss_io.convert_lng_positive_in_west(
            LONGITUDES_DEG)
        self.assertTrue(numpy.allclose(lng_positive_in_west_deg,
                                       EXPECTED_LNG_POSITIVE_IN_WEST_DEG,
                                       atol=TOLERANCE))

    def test_rowcol_to_latlng(self):
        """Ensures correct output from rowcol_to_latlng."""

        (grid_latitudes_deg, grid_longitudes_deg) = myrorss_io.rowcol_to_latlng(
            GRID_ROWS, GRID_COLUMNS,
            nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
            nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
            lat_spacing_deg=LAT_SPACING_DEG, lng_spacing_deg=LNG_SPACING_DEG)

        self.assertTrue(
            numpy.allclose(grid_latitudes_deg, EXPECTED_GRID_LAT_DEG,
                           atol=TOLERANCE))
        self.assertTrue(
            numpy.allclose(grid_longitudes_deg, EXPECTED_GRID_LNG_DEG,
                           atol=TOLERANCE))

    def test_latlng_to_rowcol(self):
        """Ensures correct output from latlng_to_rowcol."""

        (these_grid_rows, these_grid_columns) = myrorss_io.latlng_to_rowcol(
            EXPECTED_GRID_LAT_DEG, EXPECTED_GRID_LNG_DEG,
            nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
            nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
            lat_spacing_deg=LAT_SPACING_DEG, lng_spacing_deg=LNG_SPACING_DEG)

        self.assertTrue(
            numpy.allclose(these_grid_rows, GRID_ROWS, atol=TOLERANCE))
        self.assertTrue(
            numpy.allclose(these_grid_columns, GRID_COLUMNS, atol=TOLERANCE))

    def test_get_center_of_grid(self):
        """Ensures correct output from get_center_of_grid."""

        (this_center_lat_deg,
         this_center_lng_deg) = myrorss_io.get_center_of_grid(
            nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
            nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
            lat_spacing_deg=LAT_SPACING_DEG, lng_spacing_deg=LNG_SPACING_DEG,
            num_lat_in_grid=NUM_LAT_IN_GRID, num_lng_in_grid=NUM_LNG_IN_GRID)

        self.assertTrue(numpy.allclose(numpy.array([this_center_lat_deg]),
                                       numpy.array([CENTER_LAT_IN_GRID_DEG]),
                                       atol=TOLERANCE))
        self.assertTrue(numpy.allclose(numpy.array([this_center_lng_deg]),
                                       numpy.array([CENTER_LNG_IN_GRID_DEG]),
                                       atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
