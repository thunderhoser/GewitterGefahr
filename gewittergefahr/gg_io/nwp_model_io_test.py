"""Unit tests for nwp_model_io.py."""

import unittest
from gewittergefahr.gg_io import nwp_model_io
from gewittergefahr.gg_utils import rap_model_utils

LEAD_TIME_HOURS_1DIGIT = 9
LEAD_TIME_STRING_1DIGIT = '009'
LEAD_TIME_HOURS_2DIGITS = 12
LEAD_TIME_STRING_2DIGITS = '012'
LEAD_TIME_HOURS_3DIGITS = 144
LEAD_TIME_STRING_3DIGITS = '144'

FAKE_MODEL_NAME = 'foo'
FAKE_GRID_ID = '9999'

MODEL_ID_RAP130 = 'rap_130'
MODEL_ID_RAP252 = 'rap_252'
MODEL_ID_NARR = 'narr-a_221'

INIT_TIME_UNIX_SEC = 1505962800  # 0300 UTC 21 Sep 2017
LEAD_TIME_HOURS = 7

MODEL_NAME_FOR_FILES = 'rap'
GRID_ID_FOR_FILES = '130'
GRIB_TYPE_FOR_FILES = 'grib2'
PATHLESS_GRIB_FILE_NAME = 'rap_130_20170921_0300_007.grb2'

TOP_GRIB_DIRECTORY_NAME = 'grib_files'
GRIB_FILE_NAME = 'grib_files/201709/rap_130_20170921_0300_007.grb2'

GRIB1_FIELD_NAME = 'HGT:500 mb'
PATHLESS_SINGLE_FIELD_FILE_NAME = 'rap_130_20170921_0300_007_HGT:500mb.txt'

TOP_SINGLE_FIELD_DIR_NAME = 'single_field_files'
SINGLE_FIELD_FILE_NAME = (
    'single_field_files/201709/rap_130_20170921_0300_007_HGT:500mb.txt')


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

    def test_check_model_name_rap(self):
        """Ensures correct output from _check_model_name.

        In this case, model is RAP.
        """

        nwp_model_io._check_model_name(nwp_model_io.RAP_MODEL_NAME)

    def test_check_model_name_narr(self):
        """Ensures correct output from _check_model_name.

        In this case, model is NARR.
        """

        nwp_model_io._check_model_name(nwp_model_io.NARR_MODEL_NAME)

    def test_check_model_name_fake(self):
        """Ensures correct output from _check_model_name.

        In this case, model is unrecognized.
        """

        with self.assertRaises(ValueError):
            nwp_model_io._check_model_name(FAKE_MODEL_NAME)

    def test_get_model_id_for_grib_file_names_rap130(self):
        """Ensures correct output from _get_model_id_for_grib_file_names.

        In this case, model is RAP on the 130 grid.
        """

        this_model_id = nwp_model_io._get_model_id_for_grib_file_names(
            nwp_model_io.RAP_MODEL_NAME, rap_model_utils.ID_FOR_130GRID)
        self.assertTrue(this_model_id == MODEL_ID_RAP130)

    def test_get_model_id_for_grib_file_names_rap252(self):
        """Ensures correct output from _get_model_id_for_grib_file_names.

        In this case, model is RAP on the 252 grid.
        """

        this_model_id = nwp_model_io._get_model_id_for_grib_file_names(
            nwp_model_io.RAP_MODEL_NAME, rap_model_utils.ID_FOR_252GRID)
        self.assertTrue(this_model_id == MODEL_ID_RAP252)

    def test_get_model_id_for_grib_file_names_narr(self):
        """Ensures correct output from _get_model_id_for_grib_file_names.

        In this case, model is NARR.
        """

        this_model_id = nwp_model_io._get_model_id_for_grib_file_names(
            nwp_model_io.NARR_MODEL_NAME)
        self.assertTrue(this_model_id == MODEL_ID_NARR)

    def test_get_pathless_grib_file_name(self):
        """Ensures correct output from _get_pathless_grib_file_name."""

        this_pathless_grib_file_name = (
            nwp_model_io._get_pathless_grib_file_name(
                INIT_TIME_UNIX_SEC, LEAD_TIME_HOURS,
                model_name=MODEL_NAME_FOR_FILES, grid_id=GRID_ID_FOR_FILES,
                grib_type=GRIB_TYPE_FOR_FILES))

        self.assertTrue(this_pathless_grib_file_name == PATHLESS_GRIB_FILE_NAME)

    def test_get_pathless_single_field_file_name(self):
        """Ensures correct output from _get_pathless_single_field_file_name."""

        this_pathless_file_name = (
            nwp_model_io._get_pathless_single_field_file_name(
                INIT_TIME_UNIX_SEC, LEAD_TIME_HOURS,
                model_name=MODEL_NAME_FOR_FILES, grid_id=GRID_ID_FOR_FILES,
                grib1_field_name=GRIB1_FIELD_NAME))

        self.assertTrue(
            this_pathless_file_name == PATHLESS_SINGLE_FIELD_FILE_NAME)

    def test_check_grid_id_narr(self):
        """Ensures correct output from check_grid_id.

        In this case, model is NARR.
        """

        nwp_model_io.check_grid_id(nwp_model_io.NARR_MODEL_NAME)

    def test_check_grid_id_rap130(self):
        """Ensures correct output from check_grid_id.

        In this case, model is RAP on the 130 grid.
        """

        nwp_model_io.check_grid_id(nwp_model_io.RAP_MODEL_NAME,
                                   rap_model_utils.ID_FOR_130GRID)

    def test_check_grid_id_rap252(self):
        """Ensures correct output from check_grid_id.

        In this case, model is RAP on the 252 grid.
        """

        nwp_model_io.check_grid_id(nwp_model_io.RAP_MODEL_NAME,
                                   rap_model_utils.ID_FOR_252GRID)

    def test_check_grid_id_fake_grid(self):
        """Ensures correct output from check_grid_id.

        In this case, grid ID is not recognized.
        """

        with self.assertRaises(ValueError):
            nwp_model_io.check_grid_id(nwp_model_io.RAP_MODEL_NAME,
                                       FAKE_GRID_ID)

    def test_find_grib_file(self):
        """Ensures correct output from find_grib_file."""

        this_grib_file_name = nwp_model_io.find_grib_file(
            INIT_TIME_UNIX_SEC, LEAD_TIME_HOURS,
            top_directory_name=TOP_GRIB_DIRECTORY_NAME,
            model_name=MODEL_NAME_FOR_FILES, grid_id=GRID_ID_FOR_FILES,
            grib_type=GRIB_TYPE_FOR_FILES, raise_error_if_missing=False)

        self.assertTrue(this_grib_file_name == GRIB_FILE_NAME)

    def test_find_single_field_file(self):
        """Ensures correct output from find_single_field_file."""

        this_single_field_file_name = nwp_model_io.find_single_field_file(
            INIT_TIME_UNIX_SEC, LEAD_TIME_HOURS,
            top_directory_name=TOP_SINGLE_FIELD_DIR_NAME,
            model_name=MODEL_NAME_FOR_FILES, grid_id=GRID_ID_FOR_FILES,
            grib1_field_name=GRIB1_FIELD_NAME, raise_error_if_missing=False)

        self.assertTrue(this_single_field_file_name == SINGLE_FIELD_FILE_NAME)


if __name__ == '__main__':
    unittest.main()
