"""Unit tests for nwp_model_io.py."""

import unittest
from gewittergefahr.gg_io import nwp_model_io

UNIX_TIME_SEC = 1505884620  # 0517 UTC 20 Sep 2017
EXPECTED_MONTH_STRING = '201709'
EXPECTED_DATE_STRING = '20170920'
EXPECTED_HOUR_STRING = '20170920_0500'

LEAD_TIME_HOURS_1DIGIT = 9
EXPECTED_LEAD_TIME_STRING_1DIGIT = '009'
LEAD_TIME_HOURS_2DIGITS = 12
EXPECTED_LEAD_TIME_STRING_2DIGITS = '012'
LEAD_TIME_HOURS_3DIGITS = 144
EXPECTED_LEAD_TIME_STRING_3DIGITS = '144'

INIT_TIME_UNIX_SEC = 1505962800  # 0300 UTC 21 Sep 2017
LEAD_TIME_HOURS = 7
MODEL_ID_FOR_FILE_NAMES = 'ef12_hypercane'
RAW_FILE_EXTENSION = 'grb2'
EXPECTED_PATHLESS_RAW_FILE_NAME = 'ef12_hypercane_20170921_0300_007.grb2'

TOP_DIRECTORY_NAME_FOR_RAW_FILES = 'ef12_hypercane/grb2'
EXPECTED_RAW_FILE_NAME = (
    'ef12_hypercane/grb2/201709/ef12_hypercane_20170921_0300_007.grb2')

VARIABLE_NAME = 'HGT:500 mb'
EXPECTED_PATHLESS_TEXT_FILE_NAME = (
    'ef12_hypercane_20170921_0300_007_HGT:500mb.txt')

TOP_DIRECTORY_NAME_FOR_TEXT_FILES = 'ef12_hypercane/text'
EXPECTED_TEXT_FILE_NAME = (
    'ef12_hypercane/text/201709/ef12_hypercane_20170921_0300_007_HGT:500mb.txt')


class NwpModelIoTests(unittest.TestCase):
    """Each method is a unit test for nwp_model_io.py."""

    def test_time_unix_sec_to_month_string(self):
        """Ensures correct output from time_unix_sec_to_month_string."""

        this_month_string = nwp_model_io._time_unix_sec_to_month_string(
            UNIX_TIME_SEC)
        self.assertTrue(this_month_string == EXPECTED_MONTH_STRING)

    def test_time_unix_sec_to_date_string(self):
        """Ensures correct output from time_unix_sec_to_date_string."""

        this_date_string = nwp_model_io._time_unix_sec_to_date_string(
            UNIX_TIME_SEC)
        self.assertTrue(this_date_string == EXPECTED_DATE_STRING)

    def test_time_unix_sec_to_hour_string(self):
        """Ensures correct output from time_unix_sec_to_hour_string."""

        this_hour_string = nwp_model_io._time_unix_sec_to_hour_string(
            UNIX_TIME_SEC)
        self.assertTrue(this_hour_string == EXPECTED_HOUR_STRING)

    def test_lead_time_hours_to_string_1digit(self):
        """Ensures correct output from lead_time_hours_to_string.

        In this case, lead time has 1 digit.
        """

        this_lead_time_string = nwp_model_io._lead_time_hours_to_string(
            LEAD_TIME_HOURS_1DIGIT)
        self.assertTrue(
            this_lead_time_string == EXPECTED_LEAD_TIME_STRING_1DIGIT)

    def test_lead_time_hours_to_string_2digits(self):
        """Ensures correct output from lead_time_hours_to_string.

        In this case, lead time has 2 digits.
        """

        this_lead_time_string = nwp_model_io._lead_time_hours_to_string(
            LEAD_TIME_HOURS_2DIGITS)
        self.assertTrue(
            this_lead_time_string == EXPECTED_LEAD_TIME_STRING_2DIGITS)

    def test_lead_time_hours_to_string_3digits(self):
        """Ensures correct output from lead_time_hours_to_string.

        In this case, lead time has 3 digits.
        """

        this_lead_time_string = nwp_model_io._lead_time_hours_to_string(
            LEAD_TIME_HOURS_3DIGITS)
        self.assertTrue(
            this_lead_time_string == EXPECTED_LEAD_TIME_STRING_3DIGITS)

    def test_get_pathless_raw_file_name(self):
        """Ensures correct output from _get_pathless_raw_file_name."""

        pathless_raw_file_name = nwp_model_io._get_pathless_raw_file_name(
            INIT_TIME_UNIX_SEC, LEAD_TIME_HOURS,
            model_id=MODEL_ID_FOR_FILE_NAMES, file_extension=RAW_FILE_EXTENSION)

        self.assertTrue(
            pathless_raw_file_name == EXPECTED_PATHLESS_RAW_FILE_NAME)

    def test_get_pathless_text_file_name(self):
        """Ensures correct output from _get_pathless_text_file_name."""

        pathless_text_file_name = nwp_model_io._get_pathless_text_file_name(
            INIT_TIME_UNIX_SEC, LEAD_TIME_HOURS,
            model_id=MODEL_ID_FOR_FILE_NAMES, variable_name=VARIABLE_NAME)

        self.assertTrue(
            pathless_text_file_name == EXPECTED_PATHLESS_TEXT_FILE_NAME)

    def test_find_local_raw_file(self):
        """Ensures correct output from find_local_raw_file."""

        raw_file_name = nwp_model_io.find_local_raw_file(
            INIT_TIME_UNIX_SEC, LEAD_TIME_HOURS,
            top_directory_name=TOP_DIRECTORY_NAME_FOR_RAW_FILES,
            model_id_for_pathless_file_name=MODEL_ID_FOR_FILE_NAMES,
            file_extension=RAW_FILE_EXTENSION, raise_error_if_missing=False)

        self.assertTrue(raw_file_name == EXPECTED_RAW_FILE_NAME)

    def test_find_local_text_file(self):
        """Ensures correct output from find_local_text_file."""

        text_file_name = nwp_model_io.find_local_text_file(
            INIT_TIME_UNIX_SEC, LEAD_TIME_HOURS,
            top_directory_name=TOP_DIRECTORY_NAME_FOR_TEXT_FILES,
            model_id_for_pathless_file_name=MODEL_ID_FOR_FILE_NAMES,
            variable_name=VARIABLE_NAME, raise_error_if_missing=False)

        self.assertTrue(text_file_name == EXPECTED_TEXT_FILE_NAME)


if __name__ == '__main__':
    unittest.main()
