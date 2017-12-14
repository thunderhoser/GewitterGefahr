"""Unit tests for ok_mesonet_io.py."""

import unittest
from gewittergefahr.gg_io import ok_mesonet_io

DATE_STRING = '21 2017 09 15 00 00 00'
UNIX_DATE_SEC = 1505433600

UNIX_TIME_SEC = 1506195300  # 1935 UTC 23 Sep 2017
TOP_RAW_DIRECTORY_NAME = 'ok_mesonet/raw_files'
RAW_FILE_NAME = 'ok_mesonet/raw_files/2017/09/23/201709231935.mdf'


class OkMesonetIoTests(unittest.TestCase):
    """Each method is a unit test for ok_mesonet_io.py."""

    def test_date_string_to_unix_sec(self):
        """Ensures correct output from _date_string_to_unix_sec."""

        this_date_unix_sec = ok_mesonet_io._date_string_to_unix_sec(DATE_STRING)
        self.assertTrue(this_date_unix_sec == UNIX_DATE_SEC)

    def test_find_local_raw_file(self):
        """Ensures correct output from find_local_raw_file."""

        this_file_name = ok_mesonet_io.find_local_raw_file(
            unix_time_sec=UNIX_TIME_SEC,
            top_directory_name=TOP_RAW_DIRECTORY_NAME,
            raise_error_if_missing=False)
        self.assertTrue(this_file_name == RAW_FILE_NAME)


if __name__ == '__main__':
    unittest.main()
