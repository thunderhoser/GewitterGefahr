"""Unit tests for nwp_model_io.py."""

import unittest
from gewittergefahr.gg_io import nwp_model_io
from gewittergefahr.gg_utils import nwp_model_utils

LEAD_TIME_HOURS_1DIGIT = 9
LEAD_TIME_STRING_1DIGIT = '009'
LEAD_TIME_HOURS_2DIGITS = 12
LEAD_TIME_STRING_2DIGITS = '012'
LEAD_TIME_HOURS_3DIGITS = 144
LEAD_TIME_STRING_3DIGITS = '144'

PATHLESS_FILE_NAME_PREFIXES_NARR = ['narr-a_221']
PATHLESS_FILE_NAME_PREFIXES_RAP130 = ['rap_130']
PATHLESS_FILE_NAME_PREFIXES_RAP252 = ['rap_252']
PATHLESS_FILE_NAME_PREFIXES_RUC130 = ['ruc2_130', 'ruc2anl_130']
PATHLESS_FILE_NAME_PREFIXES_RUC252 = ['ruc2_252', 'ruc2anl_252']
PATHLESS_FILE_NAME_PREFIXES_RUC236 = ['ruc2_236', 'ruc2anl_236']

INIT_TIME_UNIX_SEC = 1505962800  # 0300 UTC 21 Sep 2017
LEAD_TIME_HOURS = 7
TOP_GRIB_DIRECTORY_NAME = 'grib_files'

PATHLESS_GRIB_FILE_NAMES_NARR = ['narr-a_221_20170921_0300_000.grb']
PATHLESS_GRIB_FILE_NAMES_RAP130 = ['rap_130_20170921_0300_007.grb2']
PATHLESS_GRIB_FILE_NAMES_RUC252 = [
    'ruc2_252_20170921_0300_007.grb', 'ruc2_252_20170921_0300_007.grb2',
    'ruc2anl_252_20170921_0300_007.grb', 'ruc2anl_252_20170921_0300_007.grb2']

GRIB_FILE_NAME_NARR = 'grib_files/201709/narr-a_221_20170921_0300_000.grb'
GRIB_FILE_NAME_RAP130 = 'grib_files/201709/rap_130_20170921_0300_007.grb2'
GRIB_FILE_NAME_RUC252 = 'grib_files/201709/ruc2_252_20170921_0300_007.grb'

GRIB1_FIELD_NAME = 'HGT:500 mb'
TOP_SINGLE_FIELD_DIR_NAME = 'single_field_files'

PATHLESS_SINGLE_FIELD_FILE_NAME_NARR = (
    'narr-a_221_20170921_0300_000_HGT:500mb.txt')
PATHLESS_SINGLE_FIELD_FILE_NAME_RAP130 = (
    'rap_130_20170921_0300_007_HGT:500mb.txt')
PATHLESS_SINGLE_FIELD_FILE_NAME_RUC252 = (
    'ruc2_252_20170921_0300_007_HGT:500mb.txt')

SINGLE_FIELD_FILE_NAME_NARR = (
    'single_field_files/201709/narr-a_221_20170921_0300_000_HGT:500mb.txt')
SINGLE_FIELD_FILE_NAME_RAP130 = (
    'single_field_files/201709/rap_130_20170921_0300_007_HGT:500mb.txt')
SINGLE_FIELD_FILE_NAME_RUC252 = (
    'single_field_files/201709/ruc2_252_20170921_0300_007_HGT:500mb.txt')


class NwpModelIoTests(unittest.TestCase):
    """Each method is a unit test for nwp_model_io.py."""

    def test_lead_time_to_string_1digit(self):
        """Ensures correct output from lead_time_number_to_string.

        In this case, lead time has 1 digit.
        """

        this_lead_time_string = nwp_model_io._lead_time_to_string(
            LEAD_TIME_HOURS_1DIGIT)
        self.assertTrue(this_lead_time_string == LEAD_TIME_STRING_1DIGIT)

    def test_lead_time_to_string_2digits(self):
        """Ensures correct output from lead_time_number_to_string.

        In this case, lead time has 2 digits.
        """

        this_lead_time_string = nwp_model_io._lead_time_to_string(
            LEAD_TIME_HOURS_2DIGITS)
        self.assertTrue(this_lead_time_string == LEAD_TIME_STRING_2DIGITS)

    def test_lead_time_to_string_3digits(self):
        """Ensures correct output from lead_time_number_to_string.

        In this case, lead time has 3 digits.
        """

        this_lead_time_string = nwp_model_io._lead_time_to_string(
            LEAD_TIME_HOURS_3DIGITS)
        self.assertTrue(this_lead_time_string == LEAD_TIME_STRING_3DIGITS)

    def test_get_prefixes_for_pathless_file_name_narr(self):
        """Ensures correct output from _get_prefixes_for_pathless_file_name.

        In this case, model is NARR.
        """

        these_prefixes = nwp_model_io._get_prefixes_for_pathless_file_name(
            nwp_model_utils.NARR_MODEL_NAME)
        self.assertTrue(
            set(these_prefixes) == set(PATHLESS_FILE_NAME_PREFIXES_NARR))

    def test_get_prefixes_for_pathless_file_name_rap130(self):
        """Ensures correct output from _get_prefixes_for_pathless_file_name.

        In this case, model is RAP on the 130 grid.
        """

        these_prefixes = nwp_model_io._get_prefixes_for_pathless_file_name(
            nwp_model_utils.RAP_MODEL_NAME, nwp_model_utils.ID_FOR_130GRID)
        self.assertTrue(
            set(these_prefixes) == set(PATHLESS_FILE_NAME_PREFIXES_RAP130))

    def test_get_prefixes_for_pathless_file_name_rap252(self):
        """Ensures correct output from _get_prefixes_for_pathless_file_name.

        In this case, model is RAP on the 252 grid.
        """

        these_prefixes = nwp_model_io._get_prefixes_for_pathless_file_name(
            nwp_model_utils.RAP_MODEL_NAME, nwp_model_utils.ID_FOR_252GRID)
        self.assertTrue(
            set(these_prefixes) == set(PATHLESS_FILE_NAME_PREFIXES_RAP252))

    def test_get_prefixes_for_pathless_file_name_ruc130(self):
        """Ensures correct output from _get_prefixes_for_pathless_file_name.

        In this case, model is RUC on the 130 grid.
        """

        these_prefixes = nwp_model_io._get_prefixes_for_pathless_file_name(
            nwp_model_utils.RUC_MODEL_NAME, nwp_model_utils.ID_FOR_130GRID)
        self.assertTrue(
            set(these_prefixes) == set(PATHLESS_FILE_NAME_PREFIXES_RUC130))

    def test_get_prefixes_for_pathless_file_name_ruc252(self):
        """Ensures correct output from _get_prefixes_for_pathless_file_name.

        In this case, model is RUC on the 252 grid.
        """

        these_prefixes = nwp_model_io._get_prefixes_for_pathless_file_name(
            nwp_model_utils.RUC_MODEL_NAME, nwp_model_utils.ID_FOR_252GRID)
        self.assertTrue(
            set(these_prefixes) == set(PATHLESS_FILE_NAME_PREFIXES_RUC252))

    def test_get_prefixes_for_pathless_file_name_ruc236(self):
        """Ensures correct output from _get_prefixes_for_pathless_file_name.

        In this case, model is RUC on the 236 grid.
        """

        these_prefixes = nwp_model_io._get_prefixes_for_pathless_file_name(
            nwp_model_utils.RUC_MODEL_NAME, nwp_model_utils.ID_FOR_236GRID)
        self.assertTrue(
            set(these_prefixes) == set(PATHLESS_FILE_NAME_PREFIXES_RUC236))

    def test_get_pathless_grib_file_names_narr(self):
        """Ensures correct output from _get_pathless_grib_file_names.

        In this case, model is NARR.
        """

        these_pathless_file_names = nwp_model_io._get_pathless_grib_file_names(
            INIT_TIME_UNIX_SEC, model_name=nwp_model_utils.NARR_MODEL_NAME)
        self.assertTrue(set(these_pathless_file_names) ==
                        set(PATHLESS_GRIB_FILE_NAMES_NARR))

    def test_get_pathless_grib_file_names_rap130(self):
        """Ensures correct output from _get_pathless_grib_file_names.

        In this case, model is RAP on the 130 grid.
        """

        these_pathless_file_names = nwp_model_io._get_pathless_grib_file_names(
            INIT_TIME_UNIX_SEC, lead_time_hours=LEAD_TIME_HOURS,
            model_name=nwp_model_utils.RAP_MODEL_NAME,
            grid_id=nwp_model_utils.ID_FOR_130GRID)

        self.assertTrue(set(these_pathless_file_names) ==
                        set(PATHLESS_GRIB_FILE_NAMES_RAP130))

    def test_get_pathless_grib_file_names_ruc252(self):
        """Ensures correct output from _get_pathless_grib_file_names.

        In this case, model is RUC on the 252 grid.
        """

        these_pathless_file_names = nwp_model_io._get_pathless_grib_file_names(
            INIT_TIME_UNIX_SEC, lead_time_hours=LEAD_TIME_HOURS,
            model_name=nwp_model_utils.RUC_MODEL_NAME,
            grid_id=nwp_model_utils.ID_FOR_252GRID)

        self.assertTrue(set(these_pathless_file_names) ==
                        set(PATHLESS_GRIB_FILE_NAMES_RUC252))

    def test_get_pathless_single_field_file_name_narr(self):
        """Ensures correct output from _get_pathless_single_field_file_name.

        In this case, model is NARR.
        """

        this_pathless_file_name = (
            nwp_model_io._get_pathless_single_field_file_name(
                INIT_TIME_UNIX_SEC, model_name=nwp_model_utils.NARR_MODEL_NAME,
                grib1_field_name=GRIB1_FIELD_NAME))

        self.assertTrue(
            this_pathless_file_name == PATHLESS_SINGLE_FIELD_FILE_NAME_NARR)

    def test_get_pathless_single_field_file_name_rap130(self):
        """Ensures correct output from _get_pathless_single_field_file_name.

        In this case, model is RAP on the 130 grid.
        """

        this_pathless_file_name = (
            nwp_model_io._get_pathless_single_field_file_name(
                INIT_TIME_UNIX_SEC, lead_time_hours=LEAD_TIME_HOURS,
                model_name=nwp_model_utils.RAP_MODEL_NAME,
                grid_id=nwp_model_utils.ID_FOR_130GRID,
                grib1_field_name=GRIB1_FIELD_NAME))

        self.assertTrue(
            this_pathless_file_name == PATHLESS_SINGLE_FIELD_FILE_NAME_RAP130)

    def test_get_pathless_single_field_file_name_ruc252(self):
        """Ensures correct output from _get_pathless_single_field_file_name.

        In this case, model is RUC on the 252 grid.
        """

        this_pathless_file_name = (
            nwp_model_io._get_pathless_single_field_file_name(
                INIT_TIME_UNIX_SEC, lead_time_hours=LEAD_TIME_HOURS,
                model_name=nwp_model_utils.RUC_MODEL_NAME,
                grid_id=nwp_model_utils.ID_FOR_252GRID,
                grib1_field_name=GRIB1_FIELD_NAME))

        self.assertTrue(
            this_pathless_file_name == PATHLESS_SINGLE_FIELD_FILE_NAME_RUC252)

    def test_find_grib_file_narr(self):
        """Ensures correct output from find_grib_file when model is NARR."""

        this_grib_file_name = nwp_model_io.find_grib_file(
            INIT_TIME_UNIX_SEC, model_name=nwp_model_utils.NARR_MODEL_NAME,
            top_directory_name=TOP_GRIB_DIRECTORY_NAME,
            raise_error_if_missing=False)
        self.assertTrue(this_grib_file_name == GRIB_FILE_NAME_NARR)

    def test_find_grib_file_rap130(self):
        """Ensures correct output from find_grib_file.

        In this case, model is RAP on the 130 grid.
        """

        this_grib_file_name = nwp_model_io.find_grib_file(
            INIT_TIME_UNIX_SEC, lead_time_hours=LEAD_TIME_HOURS,
            model_name=nwp_model_utils.RAP_MODEL_NAME,
            grid_id=nwp_model_utils.ID_FOR_130GRID,
            top_directory_name=TOP_GRIB_DIRECTORY_NAME,
            raise_error_if_missing=False)
        self.assertTrue(this_grib_file_name == GRIB_FILE_NAME_RAP130)

    def test_find_grib_file_ruc252(self):
        """Ensures correct output from find_grib_file.

        In this case, model is RUC on the 252 grid.
        """

        this_grib_file_name = nwp_model_io.find_grib_file(
            INIT_TIME_UNIX_SEC, lead_time_hours=LEAD_TIME_HOURS,
            model_name=nwp_model_utils.RUC_MODEL_NAME,
            grid_id=nwp_model_utils.ID_FOR_252GRID,
            top_directory_name=TOP_GRIB_DIRECTORY_NAME,
            raise_error_if_missing=False)
        self.assertTrue(this_grib_file_name == GRIB_FILE_NAME_RUC252)

    def test_find_single_field_file_narr(self):
        """Ensures correct output from find_single_field_file.

        In this case, model is NARR.
        """

        this_single_field_file_name = nwp_model_io.find_single_field_file(
            INIT_TIME_UNIX_SEC, model_name=nwp_model_utils.NARR_MODEL_NAME,
            grib1_field_name=GRIB1_FIELD_NAME,
            top_directory_name=TOP_SINGLE_FIELD_DIR_NAME,
            raise_error_if_missing=False)

        self.assertTrue(
            this_single_field_file_name == SINGLE_FIELD_FILE_NAME_NARR)

    def test_find_single_field_file_rap130(self):
        """Ensures correct output from find_single_field_file.

        In this case, model is RAP on the 130 grid.
        """

        this_single_field_file_name = nwp_model_io.find_single_field_file(
            INIT_TIME_UNIX_SEC, lead_time_hours=LEAD_TIME_HOURS,
            model_name=nwp_model_utils.RAP_MODEL_NAME,
            grid_id=nwp_model_utils.ID_FOR_130GRID,
            grib1_field_name=GRIB1_FIELD_NAME,
            top_directory_name=TOP_SINGLE_FIELD_DIR_NAME,
            raise_error_if_missing=False)

        self.assertTrue(
            this_single_field_file_name == SINGLE_FIELD_FILE_NAME_RAP130)

    def test_find_single_field_file_ruc252(self):
        """Ensures correct output from find_single_field_file.

        In this case, model is RUC on the 252 grid.
        """

        this_single_field_file_name = nwp_model_io.find_single_field_file(
            INIT_TIME_UNIX_SEC, lead_time_hours=LEAD_TIME_HOURS,
            model_name=nwp_model_utils.RUC_MODEL_NAME,
            grid_id=nwp_model_utils.ID_FOR_252GRID,
            grib1_field_name=GRIB1_FIELD_NAME,
            top_directory_name=TOP_SINGLE_FIELD_DIR_NAME,
            raise_error_if_missing=False)

        self.assertTrue(
            this_single_field_file_name == SINGLE_FIELD_FILE_NAME_RUC252)


if __name__ == '__main__':
    unittest.main()
