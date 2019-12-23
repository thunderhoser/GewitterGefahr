"""Unit tests for conus_boundary.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import conus_boundary

QUERY_LATITUDES_DEG = numpy.array([
    33.7, 42.6, 39.7, 34.9, 40.2, 33.6, 36.4, 35.1, 30.8, 47.4, 44.2, 45.1,
    49.6, 38.9, 35.0, 38.1, 40.7, 47.1, 30.2, 39.2
])
QUERY_LONGITUDES_DEG = numpy.array([
    276.3, 282.7, 286.6, 287.5, 271.0, 266.4, 258.3, 257.3, 286.8, 235.0, 273.5,
    262.5, 277.2, 255.3, 271.8, 254.3, 262.1, 247.8, 262.9, 251.6
])

IN_CONUS_FLAGS = numpy.array(
    [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1], dtype=bool
)


class ConusBoundaryTests(unittest.TestCase):
    """Each method is a unit test for conus_boundary.py."""

    def test_find_points_in_conus_no_shortcuts(self):
        """Ensures correct output from find_points_in_conus.

        In this case, does not use shortcuts.
        """

        conus_latitudes_deg, conus_longitudes_deg = (
            conus_boundary.read_from_netcdf()
        )

        these_flags = conus_boundary.find_points_in_conus(
            conus_latitudes_deg=conus_latitudes_deg,
            conus_longitudes_deg=conus_longitudes_deg,
            query_latitudes_deg=QUERY_LATITUDES_DEG,
            query_longitudes_deg=QUERY_LONGITUDES_DEG, use_shortcuts=False)

        self.assertTrue(numpy.array_equal(these_flags, IN_CONUS_FLAGS))

    def test_find_points_in_conus_with_shortcuts(self):
        """Ensures correct output from find_points_in_conus.

        In this case, uses shortcuts.
        """

        conus_latitudes_deg, conus_longitudes_deg = (
            conus_boundary.read_from_netcdf()
        )

        these_flags = conus_boundary.find_points_in_conus(
            conus_latitudes_deg=conus_latitudes_deg,
            conus_longitudes_deg=conus_longitudes_deg,
            query_latitudes_deg=QUERY_LATITUDES_DEG,
            query_longitudes_deg=QUERY_LONGITUDES_DEG, use_shortcuts=True)

        self.assertTrue(numpy.array_equal(these_flags, IN_CONUS_FLAGS))


if __name__ == '__main__':
    unittest.main()
