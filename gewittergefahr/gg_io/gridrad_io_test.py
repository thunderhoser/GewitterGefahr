"""Unit tests for gridrad_io.py."""

import unittest
from gewittergefahr.gg_io import gridrad_io

GRIDRAD_TIME_SEC = 512395800
UNIX_TIME_SEC = 1490703000


class GridradIoTests(unittest.TestCase):
    """Each method is a unit test for gridrad_io.py."""

    def test_time_from_gridrad_to_unix(self):
        """Ensures correct output from _time_from_gridrad_to_unix."""

        this_time_unix_sec = gridrad_io._time_from_gridrad_to_unix(
            GRIDRAD_TIME_SEC)
        self.assertTrue(this_time_unix_sec == UNIX_TIME_SEC)


if __name__ == '__main__':
    unittest.main()
