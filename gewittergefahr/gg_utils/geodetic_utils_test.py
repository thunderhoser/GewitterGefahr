"""Unit tests for geodetic_utils.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import geodetic_utils

DEFAULT_TOLERANCE = 1e-6
TOLERANCE_DEG = 1e-2

# The following constants are used to test find_invalid_latitudes.
LATITUDES_SOME_INVALID_DEG = numpy.array(
    [numpy.nan, -numpy.inf, -180., -100., -90., -30., 0., 45., 60., 90., 100.,
     1000., numpy.inf])
INVALID_LAT_INDICES = numpy.array([0, 1, 2, 3, 10, 11, 12], dtype=int)

# The following constants are used to test find_invalid_longitudes.
LONGITUDES_SOME_INVALID_DEG = numpy.array(
    [numpy.nan, -numpy.inf, -500., -180., -135., -90., 0., 45., 135., 180.,
     270., 360., 666., numpy.inf])
INVALID_LNG_INDICES_EITHER_SIGN = numpy.array([0, 1, 2, 12, 13], dtype=int)
INVALID_LNG_INDICES_POSITIVE_IN_WEST = numpy.array(
    [0, 1, 2, 3, 4, 5, 12, 13], dtype=int)
INVALID_LNG_INDICES_NEGATIVE_IN_WEST = numpy.array(
    [0, 1, 2, 10, 11, 12, 13], dtype=int)

# The following constants are used to test get_latlng_centroid.
POINT_LATITUDES_DEG = numpy.array(
    [20., 25., 30., numpy.nan, numpy.nan, 40., numpy.nan, 55.])
POINT_LONGITUDES_DEG = numpy.array(
    [-100., 265., 275., numpy.nan, numpy.nan, -70., numpy.nan, -60.])

CENTROID_LATITUDE_DEG = 34.
CENTROID_LONGITUDE_DEG = 278.

START_LATITUDES_DEG = numpy.array([-88., -60., -30., 0., 30., 60., 88.])
START_LONGITUDES_DEG = numpy.array([0., 60., 120., 180., 240., 300., 0.])
START_TO_END_DISTANCES_METRES = 1e4 * numpy.array([1., 2., 3., 4., 5., 6., 7.])
START_TO_END_BEARINGS_DEG = numpy.array([0., 60., 120., 180., 240., 300., 360.])
END_TO_START_BEARINGS_DEG = numpy.array([180., 240., 300., 0., 60., 120., 180.])

HALF_ROOT_TWO = numpy.sqrt(2.) / 2
HALF_ROOT_THREE = numpy.sqrt(3.) / 2
SCALAR_DISPLACEMENTS_METRES = numpy.array(
    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
STANDARD_BEARINGS_DEG = numpy.array(
    [90., 60., 45., 30., 0., 315., 270., 225., 180., 135.])
GEODETIC_BEARINGS_DEG = numpy.array(
    [0., 30., 45., 60., 90., 135., 180., 225., 270., 315.])
X_DISPLACEMENTS_METRES = numpy.array(
    [0., 0.5, HALF_ROOT_TWO, HALF_ROOT_THREE, 1., HALF_ROOT_TWO, 0.,
     -HALF_ROOT_TWO, -1., -HALF_ROOT_TWO])
Y_DISPLACEMENTS_METRES = numpy.array(
    [1., HALF_ROOT_THREE, HALF_ROOT_TWO, 0.5, 0., -HALF_ROOT_TWO, -1.,
     -HALF_ROOT_TWO, 0., HALF_ROOT_TWO])


class GeodeticUtilsTests(unittest.TestCase):
    """Each method is a unit test for geodetic_utils.py."""

    def test_find_invalid_latitudes(self):
        """Ensures correct output from find_invalid_latitudes."""

        these_invalid_indices = geodetic_utils.find_invalid_latitudes(
            LATITUDES_SOME_INVALID_DEG)
        self.assertTrue(numpy.array_equal(
            these_invalid_indices, INVALID_LAT_INDICES))

    def test_find_invalid_longitudes_either_sign(self):
        """Ensures correct output from find_invalid_longitudes.

        In this case, western-hemisphere longitudes can be either positive or
        negative.
        """

        these_invalid_indices = geodetic_utils.find_invalid_longitudes(
            LONGITUDES_SOME_INVALID_DEG,
            sign_in_western_hemisphere=geodetic_utils.EITHER_SIGN_LONGITUDE_ARG)
        self.assertTrue(numpy.array_equal(
            these_invalid_indices, INVALID_LNG_INDICES_EITHER_SIGN))

    def test_find_invalid_longitudes_positive_in_west(self):
        """Ensures correct output from find_invalid_longitudes.

        In this case, western-hemisphere longitudes must be positive.
        """

        these_invalid_indices = geodetic_utils.find_invalid_longitudes(
            LONGITUDES_SOME_INVALID_DEG,
            sign_in_western_hemisphere=geodetic_utils.POSITIVE_LONGITUDE_ARG)
        self.assertTrue(numpy.array_equal(
            these_invalid_indices, INVALID_LNG_INDICES_POSITIVE_IN_WEST))

    def test_find_invalid_longitudes_negative_in_west(self):
        """Ensures correct output from find_invalid_longitudes.

        In this case, western-hemisphere longitudes must be negative.
        """

        these_invalid_indices = geodetic_utils.find_invalid_longitudes(
            LONGITUDES_SOME_INVALID_DEG,
            sign_in_western_hemisphere=geodetic_utils.NEGATIVE_LONGITUDE_ARG)
        self.assertTrue(numpy.array_equal(
            these_invalid_indices, INVALID_LNG_INDICES_NEGATIVE_IN_WEST))

    def test_get_latlng_centroid(self):
        """Ensures correct output from get_latlng_centroid."""

        this_centroid_lat_deg, this_centroid_lng_deg = (
            geodetic_utils.get_latlng_centroid(
                POINT_LATITUDES_DEG, POINT_LONGITUDES_DEG, allow_nan=True))

        self.assertTrue(numpy.isclose(
            this_centroid_lat_deg, CENTROID_LATITUDE_DEG,
            atol=DEFAULT_TOLERANCE))
        self.assertTrue(numpy.isclose(
            this_centroid_lng_deg, CENTROID_LONGITUDE_DEG,
            atol=DEFAULT_TOLERANCE))

    def test_start_points_and_distances_and_bearings_to_endpoints(self):
        """Crrctness of start_points_and_distances_and_bearings_to_endpoints."""

        end_latitudes_deg, end_longitudes_deg = (
            geodetic_utils.start_points_and_distances_and_bearings_to_endpoints(
                start_latitudes_deg=START_LATITUDES_DEG,
                start_longitudes_deg=START_LONGITUDES_DEG,
                displacements_metres=START_TO_END_DISTANCES_METRES,
                geodetic_bearings_deg=START_TO_END_BEARINGS_DEG))

        these_start_latitudes_deg, these_start_longitudes_deg = (
            geodetic_utils.start_points_and_distances_and_bearings_to_endpoints(
                start_latitudes_deg=end_latitudes_deg,
                start_longitudes_deg=end_longitudes_deg,
                displacements_metres=START_TO_END_DISTANCES_METRES,
                geodetic_bearings_deg=END_TO_START_BEARINGS_DEG))

        self.assertTrue(numpy.allclose(
            these_start_latitudes_deg, START_LATITUDES_DEG, atol=TOLERANCE_DEG))
        self.assertTrue(numpy.allclose(
            these_start_longitudes_deg, START_LONGITUDES_DEG,
            atol=TOLERANCE_DEG))

    def test_xy_components_to_displacements_and_bearings(self):
        """Ensures crrctness of xy_components_to_displacements_and_bearings."""

        these_scalar_displacements_metres, these_geodetic_bearings_deg = (
            geodetic_utils.xy_components_to_displacements_and_bearings(
                X_DISPLACEMENTS_METRES, Y_DISPLACEMENTS_METRES))

        self.assertTrue(numpy.allclose(
            these_scalar_displacements_metres, SCALAR_DISPLACEMENTS_METRES,
            atol=DEFAULT_TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_geodetic_bearings_deg, GEODETIC_BEARINGS_DEG,
            atol=DEFAULT_TOLERANCE))

    def test_displacements_and_bearings_to_xy_components(self):
        """Ensures crrctness of displacements_and_bearings_to_xy_components."""

        these_x_displacements_metres, these_y_displacements_metres = (
            geodetic_utils.displacements_and_bearings_to_xy_components(
                SCALAR_DISPLACEMENTS_METRES, GEODETIC_BEARINGS_DEG))

        self.assertTrue(numpy.allclose(
            these_x_displacements_metres, X_DISPLACEMENTS_METRES,
            atol=DEFAULT_TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_y_displacements_metres, Y_DISPLACEMENTS_METRES,
            atol=DEFAULT_TOLERANCE))

    def test_standard_to_geodetic_angles(self):
        """Ensures correct output from standard_to_geodetic_angles."""

        these_geodetic_bearings_deg = (
            geodetic_utils.standard_to_geodetic_angles(STANDARD_BEARINGS_DEG))
        self.assertTrue(numpy.allclose(
            these_geodetic_bearings_deg, GEODETIC_BEARINGS_DEG,
            atol=DEFAULT_TOLERANCE))

    def test_geodetic_to_standard_angles(self):
        """Ensures correct output from geodetic_to_standard_angles."""

        these_standard_bearings_deg = (
            geodetic_utils.geodetic_to_standard_angles(GEODETIC_BEARINGS_DEG))
        self.assertTrue(numpy.allclose(
            these_standard_bearings_deg, STANDARD_BEARINGS_DEG,
            atol=DEFAULT_TOLERANCE))


if __name__ == '__main__':
    unittest.main()
