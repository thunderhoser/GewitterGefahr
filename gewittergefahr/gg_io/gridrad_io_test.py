"""Unit tests for gridrad_io.py."""

import unittest
from gewittergefahr.gg_io import gridrad_io

GRIDRAD_TIME_SEC = 512395800
UNIX_TIME_SEC = 1490703000  # 1210 UTC 28 Mar 2017

TOP_DIRECTORY_NAME = 'gridrad_data'
PATHLESS_FILE_NAME = 'nexrad_3d_4_1_20170328T121000Z.nc'
FULL_FILE_NAME = 'gridrad_data/2017/nexrad_3d_4_1_20170328T121000Z.nc'


class GridradIoTests(unittest.TestCase):
    """Each method is a unit test for gridrad_io.py."""

    def test_time_from_gridrad_to_unix(self):
        """Ensures correct output from _time_from_gridrad_to_unix."""

        this_time_unix_sec = gridrad_io._time_from_gridrad_to_unix(
            GRIDRAD_TIME_SEC)
        self.assertTrue(this_time_unix_sec == UNIX_TIME_SEC)

    def test_get_pathless_file_name(self):
        """Ensures correct output from _get_pathless_file_name."""

        this_pathless_file_name = gridrad_io._get_pathless_file_name(
            UNIX_TIME_SEC)
        self.assertTrue(this_pathless_file_name == PATHLESS_FILE_NAME)

    def test_find_file(self):
        """Ensures correct output from find_file."""

        this_file_name = gridrad_io.find_file(
            unix_time_sec=UNIX_TIME_SEC, top_directory_name=TOP_DIRECTORY_NAME,
            raise_error_if_missing=False)
        self.assertTrue(this_file_name == FULL_FILE_NAME)


if __name__ == '__main__':
    unittest.main()
