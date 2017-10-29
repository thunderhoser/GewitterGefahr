"""Unit tests for radar_io.py."""

import unittest
import numpy
import pandas
from gewittergefahr.gg_io import radar_io

TOLERANCE = 1e-6
FAKE_DATA_SOURCE = 'poop'

LL_SHEAR_NAME_MYRORSS = radar_io.LOW_LEVEL_SHEAR_NAME_MYRORSS
LL_SHEAR_NAME_MRMS = radar_io.LOW_LEVEL_SHEAR_NAME_MRMS
LL_SHEAR_NAME_NEW = radar_io.LOW_LEVEL_SHEAR_NAME
LL_SHEAR_NAME_NEW_FAKE = 'poop'

SHEAR_HEIGHTS_M_AGL = numpy.array([0])
NON_SHEAR_HEIGHTS_MYRORSS_M_AGL = numpy.array([250])
NON_SHEAR_HEIGHTS_MRMS_M_AGL = numpy.array([500])

REFL_HEIGHTS_M_AGL = numpy.array([500, 1000, 2000, 3000, 5000, 10000])
REFL_HEIGHTS_ONE_BAD_M_AGL = numpy.array([500, 1000, 2000, 3456, 5000, 10000])

FILE_TIME_UNIX_SEC = 1507234802  # 202002 UTC 5 Oct 2017
FILE_SPC_DATE_UNIX_SEC = 1507234802
PATHLESS_ZIPPED_FILE_NAME = '20171005-202002.netcdf.gz'
PATHLESS_UNZIPPED_FILE_NAME = '20171005-202002.netcdf'

THESE_GRID_ROWS = numpy.linspace(0, 10, num=11, dtype=int)
THESE_GRID_COLUMNS = numpy.linspace(0, 10, num=11, dtype=int)
THESE_NUM_GRID_CELLS = numpy.linspace(0, 10, num=11, dtype=int)
THESE_RADAR_VALUES = numpy.linspace(0, 10, num=11)

SENTINEL_VALUES = numpy.array([-99000., -99001.])
RADAR_FIELD_WITH_SENTINELS = radar_io.VIL_NAME
THESE_RADAR_VALUES[0] = SENTINEL_VALUES[0]
THESE_RADAR_VALUES[2] = SENTINEL_VALUES[1]
THESE_RADAR_VALUES[4] = SENTINEL_VALUES[0]
THESE_RADAR_VALUES[6] = SENTINEL_VALUES[1]

THIS_DICTIONARY = {radar_io.GRID_ROW_COLUMN: THESE_GRID_ROWS,
                   radar_io.GRID_COLUMN_COLUMN: THESE_GRID_COLUMNS,
                   radar_io.NUM_GRID_CELL_COLUMN: THESE_NUM_GRID_CELLS,
                   RADAR_FIELD_WITH_SENTINELS: THESE_RADAR_VALUES}
SPARSE_GRID_TABLE_WITH_SENTINELS = pandas.DataFrame.from_dict(THIS_DICTIONARY)

THESE_SENTINEL_INDICES = numpy.array([0, 2, 4, 6], dtype=int)
SPARSE_GRID_TABLE_NO_SENTINELS = SPARSE_GRID_TABLE_WITH_SENTINELS.drop(
    SPARSE_GRID_TABLE_WITH_SENTINELS.index[THESE_SENTINEL_INDICES], axis=0,
    inplace=False)

UNIX_TIME_1200UTC_SEC = 1506168000
UNIX_TIME_0000UTC_SEC = 1506211200
UNIX_TIME_115959UTC_SEC = 1506254399
SPC_DATE_STRING = '20170923'

RELATIVE_DIR_NAME_MYRORSS = LL_SHEAR_NAME_MYRORSS + '/00.00'
RELATIVE_DIR_NAME_MRMS = LL_SHEAR_NAME_MRMS + '/00.00'

TOP_RAW_DIRECTORY_NAME = 'radar'
RAW_FILE_NAME_MYRORSS = (
    'radar/20171005/' + LL_SHEAR_NAME_MYRORSS +
    '/00.00/20171005-202002.netcdf.gz')
RAW_FILE_NAME_MRMS = (
    'radar/20171005/' + LL_SHEAR_NAME_MRMS + '/00.00/20171005-202002.netcdf.gz')

NW_GRID_POINT_LAT_DEG = 55.
NW_GRID_POINT_LNG_DEG = 230.
LAT_SPACING_DEG = 0.01
LNG_SPACING_DEG = 0.01

GRID_ROW_INDICES = numpy.linspace(0, 10, num=11, dtype=int)
GRID_COLUMN_INDICES = numpy.linspace(0, 10, num=11, dtype=int)
GRID_POINT_LATITUDES_DEG = numpy.array(
    [55., 54.99, 54.98, 54.97, 54.96, 54.95, 54.94, 54.93, 54.92, 54.91, 54.9])
GRID_POINT_LONGITUDES_DEG = numpy.array(
    [230., 230.01, 230.02, 230.03, 230.04, 230.05, 230.06, 230.07, 230.08,
     230.09, 230.1])

NUM_GRID_ROWS = 3501
NUM_GRID_COLUMNS = 7001
GRID_CENTER_LATITUDE_DEG = 37.5
GRID_CENTER_LONGITUDE_DEG = 265.


class RadarIoTests(unittest.TestCase):
    """Each method is a unit test for radar_io.py."""

    def test_check_data_source_myrorss(self):
        """Ensures correct output from _check_data_source.

        In this case, data source is MYRORSS.
        """

        radar_io._check_data_source(radar_io.MYRORSS_SOURCE_ID)

    def test_check_data_source_mrms(self):
        """Ensures correct output from _check_data_source.

        In this case, data source is MRMS.
        """

        radar_io._check_data_source(radar_io.MRMS_SOURCE_ID)

    def test_check_data_source_fake(self):
        """Ensures correct output from _check_data_source.

        In this case, data source is fake.
        """

        with self.assertRaises(ValueError):
            radar_io._check_data_source(FAKE_DATA_SOURCE)

    def test_check_field_name_orig_myrorss_valid(self):
        """Ensures correct output from _check_field_name_orig.

        In this case, input is valid field name from MYRORSS.
        """

        radar_io._check_field_name_orig(LL_SHEAR_NAME_MYRORSS,
                                        data_source=radar_io.MYRORSS_SOURCE_ID)

    def test_check_field_name_orig_myrorss_invalid(self):
        """Ensures correct output from _check_field_name_orig.

        In this case, input is field name from MRMS (invalid for MYRORSS).
        """

        with self.assertRaises(ValueError):
            radar_io._check_field_name_orig(
                LL_SHEAR_NAME_MRMS, data_source=radar_io.MYRORSS_SOURCE_ID)

    def test_check_field_name_orig_mrms_valid(self):
        """Ensures correct output from _check_field_name_orig.

        In this case, input is valid field name from MRMS.
        """

        radar_io._check_field_name_orig(LL_SHEAR_NAME_MRMS,
                                        data_source=radar_io.MRMS_SOURCE_ID)

    def test_check_field_name_orig_mrms_invalid(self):
        """Ensures correct output from _check_field_name_orig.

        In this case, input is field name from MYRORSS (invalid for MRMS).
        """

        with self.assertRaises(ValueError):
            radar_io._check_field_name_orig(LL_SHEAR_NAME_MYRORSS,
                                            data_source=radar_io.MRMS_SOURCE_ID)

    def test_check_field_name_valid(self):
        """Ensures correct output from check_field_name.

        In this case, field name is valid.
        """

        radar_io.check_field_name(LL_SHEAR_NAME_NEW)

    def test_check_field_name_invalid(self):
        """Ensures correct output from check_field_name.

        In this case, field name is invalid.
        """

        with self.assertRaises(ValueError):
            radar_io.check_field_name(LL_SHEAR_NAME_NEW_FAKE)

    def test_field_name_orig_to_new_myrorss(self):
        """Ensures correct output from _field_name_orig_to_new.

        In this case, original field name is from MYRORSS.
        """

        this_field_name = radar_io._field_name_orig_to_new(
            LL_SHEAR_NAME_MYRORSS, data_source=radar_io.MYRORSS_SOURCE_ID)
        self.assertTrue(this_field_name == LL_SHEAR_NAME_NEW)

    def test_field_name_orig_to_new_mrms(self):
        """Ensures correct output from _field_name_orig_to_new.

        In this case, original field name is from MRMS.
        """

        this_field_name = radar_io._field_name_orig_to_new(
            LL_SHEAR_NAME_MRMS, data_source=radar_io.MRMS_SOURCE_ID)
        self.assertTrue(this_field_name == LL_SHEAR_NAME_NEW)

    def test_field_name_new_to_orig_myrorss(self):
        """Ensures correct output from _field_name_new_to_orig.

        In this case, original field name is from MYRORSS.
        """

        this_field_name_myrorss = radar_io._field_name_new_to_orig(
            LL_SHEAR_NAME_NEW, data_source=radar_io.MYRORSS_SOURCE_ID)
        self.assertTrue(this_field_name_myrorss == LL_SHEAR_NAME_MYRORSS)

    def test_field_name_new_to_orig_mrms(self):
        """Ensures correct output from _field_name_new_to_orig.

        In this case, original field name is from MRMS.
        """

        this_field_name_mrms = radar_io._field_name_new_to_orig(
            LL_SHEAR_NAME_NEW, data_source=radar_io.MRMS_SOURCE_ID)
        self.assertTrue(this_field_name_mrms == LL_SHEAR_NAME_MRMS)

    def test_get_valid_heights_for_field_shear_myrorss(self):
        """Ensures correct output from _get_valid_heights_for_field.

        In this case, the field is azimuthal shear in MYRORSS.
        """

        these_valid_heights_m_agl = radar_io._get_valid_heights_for_field(
            radar_io.MID_LEVEL_SHEAR_NAME,
            data_source=radar_io.MYRORSS_SOURCE_ID)
        self.assertTrue(
            numpy.array_equal(these_valid_heights_m_agl, SHEAR_HEIGHTS_M_AGL))

    def test_get_valid_heights_for_field_shear_mrms(self):
        """Ensures correct output from _get_valid_heights_for_field.

        In this case, the field is azimuthal shear in MRMS.
        """

        these_valid_heights_m_agl = radar_io._get_valid_heights_for_field(
            radar_io.MID_LEVEL_SHEAR_NAME, data_source=radar_io.MRMS_SOURCE_ID)
        self.assertTrue(
            numpy.array_equal(these_valid_heights_m_agl, SHEAR_HEIGHTS_M_AGL))

    def test_get_valid_heights_for_field_non_shear_myrorss(self):
        """Ensures correct output from _get_valid_heights_for_field.

        In this case, the field is a non-shear field in MYRORSS.
        """

        these_valid_heights_m_agl = radar_io._get_valid_heights_for_field(
            radar_io.REFL_M10CELSIUS_NAME,
            data_source=radar_io.MYRORSS_SOURCE_ID)
        self.assertTrue(numpy.array_equal(these_valid_heights_m_agl,
                                          NON_SHEAR_HEIGHTS_MYRORSS_M_AGL))

    def test_get_valid_heights_for_field_non_shear_mrms(self):
        """Ensures correct output from _get_valid_heights_for_field.

        In this case, the field is a non-shear field in MRMS.
        """

        these_valid_heights_m_agl = radar_io._get_valid_heights_for_field(
            radar_io.REFL_M10CELSIUS_NAME, data_source=radar_io.MRMS_SOURCE_ID)
        self.assertTrue(numpy.array_equal(these_valid_heights_m_agl,
                                          NON_SHEAR_HEIGHTS_MRMS_M_AGL))

    def test_get_valid_heights_for_field_reflectivity(self):
        """Ensures correct output from _get_valid_heights_for_field.

        In this case, the field is simple reflectivity.
        """

        these_valid_heights_m_agl = radar_io._get_valid_heights_for_field(
            radar_io.REFL_NAME, data_source=radar_io.MYRORSS_SOURCE_ID)
        self.assertTrue(len(these_valid_heights_m_agl) > 1)

    def test_check_reflectivity_heights_valid(self):
        """Ensures correct output from _check_reflectivity_heights.

        In this case, all heights are valid.
        """

        radar_io._check_reflectivity_heights(REFL_HEIGHTS_M_AGL)

    def test_check_reflectivity_heights_invalid(self):
        """Ensures correct output from _check_reflectivity_heights.

        In this case, at least one height is invalid.
        """

        with self.assertRaises(ValueError):
            radar_io._check_reflectivity_heights(REFL_HEIGHTS_ONE_BAD_M_AGL)

    def test_get_pathless_raw_file_name_zipped(self):
        """Ensures correct output from _get_pathless_raw_file_name.

        In this case, generating name for zipped file.
        """

        this_pathless_file_name = radar_io._get_pathless_raw_file_name(
            FILE_TIME_UNIX_SEC, zipped=True)
        self.assertTrue(this_pathless_file_name == PATHLESS_ZIPPED_FILE_NAME)

    def test_get_pathless_raw_file_name_unzipped(self):
        """Ensures correct output from _get_pathless_raw_file_name.

        In this case, generating name for unzipped file.
        """

        this_pathless_file_name = radar_io._get_pathless_raw_file_name(
            FILE_TIME_UNIX_SEC, zipped=False)
        self.assertTrue(this_pathless_file_name == PATHLESS_UNZIPPED_FILE_NAME)

    def test_remove_sentinels(self):
        """Ensures correct output from _remove_sentinels."""

        this_sparse_grid_table = radar_io._remove_sentinels(
            SPARSE_GRID_TABLE_WITH_SENTINELS,
            field_name=RADAR_FIELD_WITH_SENTINELS,
            sentinel_values=SENTINEL_VALUES)
        self.assertTrue(
            this_sparse_grid_table.equals(SPARSE_GRID_TABLE_NO_SENTINELS))

    def test_time_unix_sec_to_spc_date_1200utc(self):
        """Ensures correct output from time_unix_sec_to_spc_date.

        In this case, input time is 1200 UTC 23 Sep 2017 (beginning of SPC date
        "20170923").
        """

        this_spc_date_string = radar_io.time_unix_sec_to_spc_date(
            UNIX_TIME_1200UTC_SEC)
        self.assertTrue(this_spc_date_string == SPC_DATE_STRING)

    def test_time_unix_sec_to_spc_date_0000utc(self):
        """Ensures correct output from time_unix_sec_to_spc_date.

        In this case, input time is 0000 UTC 24 Sep 2017 (middle of SPC date
        "20170923").
        """

        this_spc_date_string = radar_io.time_unix_sec_to_spc_date(
            UNIX_TIME_0000UTC_SEC)
        self.assertTrue(this_spc_date_string == SPC_DATE_STRING)

    def test_time_unix_sec_to_spc_date_115959utc(self):
        """Ensures correct output from time_unix_sec_to_spc_date.

        In this case, input time is 115959 UTC 24 Sep 2017 (end of SPC date
        "20170923").
        """

        this_spc_date_string = radar_io.time_unix_sec_to_spc_date(
            UNIX_TIME_115959UTC_SEC)
        self.assertTrue(this_spc_date_string == SPC_DATE_STRING)

    def test_get_relative_dir_for_raw_files_myrorss(self):
        """Ensures correct output from get_relative_dir_for_raw_files.

        In this case, data source is MYRORSS.
        """

        this_relative_dir_name = radar_io.get_relative_dir_for_raw_files(
            field_name=LL_SHEAR_NAME_NEW,
            data_source=radar_io.MYRORSS_SOURCE_ID)
        self.assertTrue(this_relative_dir_name == RELATIVE_DIR_NAME_MYRORSS)

    def test_get_relative_dir_for_raw_files_mrms(self):
        """Ensures correct output from get_relative_dir_for_raw_files.

        In this case, data source is MRMS.
        """

        this_relative_dir_name = radar_io.get_relative_dir_for_raw_files(
            field_name=LL_SHEAR_NAME_NEW,
            data_source=radar_io.MRMS_SOURCE_ID)
        self.assertTrue(this_relative_dir_name == RELATIVE_DIR_NAME_MRMS)

    def test_find_raw_file_myrorss(self):
        """Ensures correct output from find_raw_file."""

        this_raw_file_name = radar_io.find_raw_file(
            unix_time_sec=FILE_TIME_UNIX_SEC,
            spc_date_unix_sec=FILE_SPC_DATE_UNIX_SEC,
            field_name=LL_SHEAR_NAME_NEW,
            data_source=radar_io.MYRORSS_SOURCE_ID,
            top_directory_name=TOP_RAW_DIRECTORY_NAME,
            raise_error_if_missing=False)
        self.assertTrue(this_raw_file_name == RAW_FILE_NAME_MYRORSS)

    def test_find_raw_file_mrms(self):
        """Ensures correct output from find_raw_file."""

        this_raw_file_name = radar_io.find_raw_file(
            unix_time_sec=FILE_TIME_UNIX_SEC,
            spc_date_unix_sec=FILE_SPC_DATE_UNIX_SEC,
            field_name=LL_SHEAR_NAME_NEW,
            data_source=radar_io.MRMS_SOURCE_ID,
            top_directory_name=TOP_RAW_DIRECTORY_NAME,
            raise_error_if_missing=False)
        self.assertTrue(this_raw_file_name == RAW_FILE_NAME_MRMS)

    def test_rowcol_to_latlng(self):
        """Ensures correct output from rowcol_to_latlng."""

        (these_latitudes_deg,
         these_longitudes_deg) = radar_io.rowcol_to_latlng(
             GRID_ROW_INDICES, GRID_COLUMN_INDICES,
             nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
             nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
             lat_spacing_deg=LAT_SPACING_DEG, lng_spacing_deg=LNG_SPACING_DEG)

        self.assertTrue(numpy.allclose(
            these_latitudes_deg, GRID_POINT_LATITUDES_DEG, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_longitudes_deg, GRID_POINT_LONGITUDES_DEG, atol=TOLERANCE))

    def test_latlng_to_rowcol(self):
        """Ensures correct output from latlng_to_rowcol."""

        (these_row_indices, these_column_indices) = radar_io.latlng_to_rowcol(
            GRID_POINT_LATITUDES_DEG, GRID_POINT_LONGITUDES_DEG,
            nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
            nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
            lat_spacing_deg=LAT_SPACING_DEG, lng_spacing_deg=LNG_SPACING_DEG)

        self.assertTrue(numpy.allclose(
            these_row_indices, GRID_ROW_INDICES, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_column_indices, GRID_COLUMN_INDICES, atol=TOLERANCE))

    def test_get_center_of_grid(self):
        """Ensures correct output from get_center_of_grid."""

        (this_center_lat_deg,
         this_center_lng_deg) = radar_io.get_center_of_grid(
             nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
             nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
             lat_spacing_deg=LAT_SPACING_DEG, lng_spacing_deg=LNG_SPACING_DEG,
             num_grid_rows=NUM_GRID_ROWS, num_grid_columns=NUM_GRID_COLUMNS)

        self.assertTrue(numpy.isclose(
            this_center_lat_deg, GRID_CENTER_LATITUDE_DEG, atol=TOLERANCE))
        self.assertTrue(numpy.isclose(
            this_center_lng_deg, GRID_CENTER_LONGITUDE_DEG, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
