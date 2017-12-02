"""Unit tests for gridrad_io.py."""

import unittest
from gewittergefahr.gg_io import gridrad_io

DIFFERENTIAL_REFL_NAME = gridrad_io.DIFFERENTIAL_REFL_NAME
DIFFERENTIAL_REFL_NAME_ORIG = gridrad_io.DIFFERENTIAL_REFL_NAME_ORIG
DIFFERENTIAL_REFL_NAME_FAKE = 'foo'

GRIDRAD_TIME_SEC = 512395800
UNIX_TIME_SEC = 1490703000


class GridradIoTests(unittest.TestCase):
    """Each method is a unit test for gridrad_io.py."""

    def test_check_field_name_valid(self):
        """Ensures correct output from _check_field_name.

        In this case, name of input field is valid.
        """

        gridrad_io._check_field_name(DIFFERENTIAL_REFL_NAME)

    def test_check_field_name_invalid(self):
        """Ensures correct output from _check_field_name.

        In this case, name of input field is invalid.
        """

        with self.assertRaises(ValueError):
            gridrad_io._check_field_name(DIFFERENTIAL_REFL_NAME_FAKE)

    def test_field_name_new_to_orig(self):
        """Ensures correct output from _field_name_new_to_orig."""

        this_field_name_orig = gridrad_io._field_name_new_to_orig(
            DIFFERENTIAL_REFL_NAME)
        self.assertTrue(this_field_name_orig == DIFFERENTIAL_REFL_NAME_ORIG)

    def test_time_from_gridrad_to_unix(self):
        """Ensures correct output from _time_from_gridrad_to_unix."""

        this_time_unix_sec = gridrad_io._time_from_gridrad_to_unix(
            GRIDRAD_TIME_SEC)
        self.assertTrue(this_time_unix_sec == UNIX_TIME_SEC)


if __name__ == '__main__':
    unittest.main()
