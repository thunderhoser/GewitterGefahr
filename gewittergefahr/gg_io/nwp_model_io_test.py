"""Unit tests for nwp_model_io.py."""

import unittest
from gewittergefahr.gg_io import nwp_model_io

LEAD_TIME_HOURS_1DIGIT = 9
LEAD_TIME_STRING_1DIGIT = '009'
LEAD_TIME_HOURS_2DIGITS = 12
LEAD_TIME_STRING_2DIGITS = '012'
LEAD_TIME_HOURS_3DIGITS = 144
LEAD_TIME_STRING_3DIGITS = '144'

INIT_TIME_UNIX_SEC = 1505962800  # 0300 UTC 21 Sep 2017
LEAD_TIME_HOURS = 7
MODEL_ID_FOR_FILE_NAMES = 'ef12_hypercane'
RAW_FILE_EXTENSION = '.grb2'
PATHLESS_RAW_FILE_NAME = 'ef12_hypercane_20170921_0300_007.grb2'

TOP_DIRECTORY_NAME_FOR_RAW_FILES = 'ef12_hypercane/grb2'
RAW_FILE_NAME = (
    'ef12_hypercane/grb2/201709/ef12_hypercane_20170921_0300_007.grb2')

VARIABLE_ID = 'HGT:500 mb'
PATHLESS_TEXT_FILE_NAME = 'ef12_hypercane_20170921_0300_007_HGT:500mb.txt'

TOP_DIRECTORY_NAME_FOR_TEXT_FILES = 'ef12_hypercane/text'
TEXT_FILE_NAME = (
    'ef12_hypercane/text/201709/ef12_hypercane_20170921_0300_007_HGT:500mb.txt')


class NwpModelIoTests(unittest.TestCase):
    """Each method is a unit test for nwp_model_io.py."""

    def test_lead_time_number_to_string_1digit(self):
        """Ensures correct output from lead_time_number_to_string.

        In this case, lead time has 1 digit.
        """

        this_lead_time_string = nwp_model_io._lead_time_number_to_string(
            LEAD_TIME_HOURS_1DIGIT)
        self.assertTrue(this_lead_time_string == LEAD_TIME_STRING_1DIGIT)

    def test_lead_time_number_to_string_2digits(self):
        """Ensures correct output from lead_time_number_to_string.

        In this case, lead time has 2 digits.
        """

        this_lead_time_string = nwp_model_io._lead_time_number_to_string(
            LEAD_TIME_HOURS_2DIGITS)
        self.assertTrue(this_lead_time_string == LEAD_TIME_STRING_2DIGITS)

    def test_lead_time_number_to_string_3digits(self):
        """Ensures correct output from lead_time_number_to_string.

        In this case, lead time has 3 digits.
        """

        this_lead_time_string = nwp_model_io._lead_time_number_to_string(
            LEAD_TIME_HOURS_3DIGITS)
        self.assertTrue(this_lead_time_string == LEAD_TIME_STRING_3DIGITS)

    def test_get_pathless_raw_file_name(self):
        """Ensures correct output from _get_pathless_raw_file_name."""

        this_pathless_raw_file_name = nwp_model_io._get_pathless_raw_file_name(
            INIT_TIME_UNIX_SEC, LEAD_TIME_HOURS,
            model_id=MODEL_ID_FOR_FILE_NAMES, file_extension=RAW_FILE_EXTENSION)

        self.assertTrue(this_pathless_raw_file_name == PATHLESS_RAW_FILE_NAME)

    def test_get_pathless_text_file_name(self):
        """Ensures correct output from _get_pathless_text_file_name."""

        this_pathless_text_file_name = (
            nwp_model_io._get_pathless_text_file_name(
                INIT_TIME_UNIX_SEC, LEAD_TIME_HOURS,
                model_id=MODEL_ID_FOR_FILE_NAMES, variable_id=VARIABLE_ID))

        self.assertTrue(this_pathless_text_file_name == PATHLESS_TEXT_FILE_NAME)

    def test_find_local_raw_file(self):
        """Ensures correct output from find_local_raw_file."""

        this_raw_file_name = nwp_model_io.find_local_raw_file(
            INIT_TIME_UNIX_SEC, LEAD_TIME_HOURS,
            top_directory_name=TOP_DIRECTORY_NAME_FOR_RAW_FILES,
            model_id=MODEL_ID_FOR_FILE_NAMES, file_extension=RAW_FILE_EXTENSION,
            raise_error_if_missing=False)

        self.assertTrue(this_raw_file_name == RAW_FILE_NAME)

    def test_find_local_text_file(self):
        """Ensures correct output from find_local_text_file."""

        this_text_file_name = nwp_model_io.find_local_text_file(
            INIT_TIME_UNIX_SEC, LEAD_TIME_HOURS,
            top_directory_name=TOP_DIRECTORY_NAME_FOR_TEXT_FILES,
            model_id=MODEL_ID_FOR_FILE_NAMES, variable_id=VARIABLE_ID,
            raise_error_if_missing=False)

        self.assertTrue(this_text_file_name == TEXT_FILE_NAME)


if __name__ == '__main__':
    unittest.main()
