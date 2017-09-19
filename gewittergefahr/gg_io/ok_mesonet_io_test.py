"""Unit tests for ok_mesonet_io.py."""

import unittest
from gewittergefahr.gg_io import ok_mesonet_io

DATE_STRING = '21 2017 09 15 00 00 00'
UNIX_DATE_SEC = 1505433600


class OkMesonetIoTests(unittest.TestCase):
    """Each method is a unit test for ok_mesonet_io.py."""

    def test_date_string_to_unix_sec(self):
        """Ensures correct output from _date_string_to_unix_sec."""

        this_date_unix_sec = ok_mesonet_io._date_string_to_unix_sec(DATE_STRING)
        self.assertTrue(this_date_unix_sec == UNIX_DATE_SEC)


if __name__ == '__main__':
    unittest.main()
