"""Unit tests for geodetic_utils.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import geodetic_utils

DEFAULT_TOLERANCE = 1e-6
LATLNG_TOLERANCE_DEG = 0.01

# The following constants are used to test find_invalid_latitudes.
MESSY_LATITUDES_DEG = numpy.array(
    [numpy.nan, -numpy.inf, -180., -100., -90., -30., 0., 45., 60., 90., 100.,
     1000., numpy.inf])
INVALID_LAT_INDICES = numpy.array([0, 1, 2, 3, 10, 11, 12], dtype=int)

# The following constants are used to test find_invalid_longitudes.
MESSY_LONGITUDES_DEG = numpy.array(
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

# The following constants are used to test get_elevations.
LATITUDES_DEG = numpy.array([51.1, 53.5])
LONGITUDES_DEG = numpy.array([246, 246.5])
ELEVATIONS_M_ASL = numpy.array([1080, 675], dtype=float)

# The following constants are used to test
# start_points_and_displacements_to_endpoints.
START_LATITUDES_DEG = numpy.array([[53.5, 53.5, 53.5],
                                   [53.5, 53.5, 53.5]])
START_LONGITUDES_DEG = numpy.array([[246.5, 246.5, 246.5],
                                    [246.5, 246.5, 246.5]])
START_TO_END_DISTANCES_METRES = numpy.array([[0, 1e5, 1e5],
                                             [1e5, 1e5, 1e5]])
START_TO_END_BEARINGS_DEG = numpy.array([[45, 0, 90],
                                         [180, 270, 225]], dtype=float)

END_LATITUDES_DEG = numpy.array([[53.5, 54.398440, 53.490503],
                                 [52.601424, 53.490503, 52.859958]])
END_LONGITUDES_DEG = numpy.array([[246.5, 246.5, 248.006729],
                                  [246.5, 244.993271, 245.450152]])

# The following constants are used to test standard_to_geodetic_angles and
# geodetic_to_standard_angles.
STANDARD_BEARINGS_DEG = numpy.array([[90, 60, 45, 30, 0],
                                     [315, 270, 225, 180, 135]], dtype=float)
GEODETIC_BEARINGS_DEG = numpy.array([[0, 30, 45, 60, 90],
                                     [135, 180, 225, 270, 315]], dtype=float)

STANDARD_BEARINGS_DEG = numpy.expand_dims(STANDARD_BEARINGS_DEG, axis=0)
GEODETIC_BEARINGS_DEG = numpy.expand_dims(GEODETIC_BEARINGS_DEG, axis=0)

# The following constants are used to test
# xy_to_scalar_displacements_and_bearings and
# scalar_displacements_and_bearings_to_xy.
SCALAR_DISPLACEMENTS_METRES = numpy.array([[1, 2, 3, 4, 5],
                                           [5, 4, 3, 2, 1]], dtype=float)

HALF_ROOT_TWO = numpy.sqrt(2) / 2
HALF_ROOT_THREE = numpy.sqrt(3) / 2
X_DISPLACEMENTS_METRES = numpy.array(
    [[0, 1, 3 * HALF_ROOT_TWO, 4 * HALF_ROOT_THREE, 5],
     [5 * HALF_ROOT_TWO, 0, -3 * HALF_ROOT_TWO, -2, -HALF_ROOT_TWO]])
Y_DISPLACEMENTS_METRES = numpy.array(
    [[1, 2 * HALF_ROOT_THREE, 3 * HALF_ROOT_TWO, 2, 0],
     [-5 * HALF_ROOT_TWO, -4, -3 * HALF_ROOT_TWO, 0, HALF_ROOT_TWO]])

SCALAR_DISPLACEMENTS_METRES = numpy.expand_dims(
    SCALAR_DISPLACEMENTS_METRES, axis=0)
X_DISPLACEMENTS_METRES = numpy.expand_dims(X_DISPLACEMENTS_METRES, axis=0)
Y_DISPLACEMENTS_METRES = numpy.expand_dims(Y_DISPLACEMENTS_METRES, axis=0)

# The following constants are used to test rotate_displacement_vectors.
CCW_ROTATION_ANGLE_DEG = 90.

X_PRIME_DISPLACEMENTS_METRES = numpy.array(
    [[-1, -2 * HALF_ROOT_THREE, -3 * HALF_ROOT_TWO, -2, 0],
     [5 * HALF_ROOT_TWO, 4, 3 * HALF_ROOT_TWO, 0, -HALF_ROOT_TWO]])
Y_PRIME_DISPLACEMENTS_METRES = numpy.array(
    [[0, 1, 3 * HALF_ROOT_TWO, 4 * HALF_ROOT_THREE, 5],
     [5 * HALF_ROOT_TWO, 0, -3 * HALF_ROOT_TWO, -2, -HALF_ROOT_TWO]])

X_PRIME_DISPLACEMENTS_METRES = numpy.expand_dims(
    X_PRIME_DISPLACEMENTS_METRES, axis=0)
Y_PRIME_DISPLACEMENTS_METRES = numpy.expand_dims(
    Y_PRIME_DISPLACEMENTS_METRES, axis=0)


class GeodeticUtilsTests(unittest.TestCase):
    """Each method is a unit test for geodetic_utils.py."""

    def test_find_invalid_latitudes(self):
        """Ensures correct output from find_invalid_latitudes."""

        these_invalid_indices = geodetic_utils.find_invalid_latitudes(
            MESSY_LATITUDES_DEG)
        self.assertTrue(numpy.array_equal(
            these_invalid_indices, INVALID_LAT_INDICES))

    def test_find_invalid_longitudes_either_sign(self):
        """Ensures correct output from find_invalid_longitudes.

        In this case, longitudes in the western hemisphere can be either
        positive or negative.
        """

        these_invalid_indices = geodetic_utils.find_invalid_longitudes(
            longitudes_deg=MESSY_LONGITUDES_DEG,
            sign_in_western_hemisphere=geodetic_utils.EITHER_SIGN_LONGITUDE_ARG)
        self.assertTrue(numpy.array_equal(
            these_invalid_indices, INVALID_LNG_INDICES_EITHER_SIGN))

    def test_find_invalid_longitudes_positive_in_west(self):
        """Ensures correct output from find_invalid_longitudes.

        In this case, longitudes in the western hemisphere must be positive.
        """

        these_invalid_indices = geodetic_utils.find_invalid_longitudes(
            longitudes_deg=MESSY_LONGITUDES_DEG,
            sign_in_western_hemisphere=geodetic_utils.POSITIVE_LONGITUDE_ARG)
        self.assertTrue(numpy.array_equal(
            these_invalid_indices, INVALID_LNG_INDICES_POSITIVE_IN_WEST))

    def test_find_invalid_longitudes_negative_in_west(self):
        """Ensures correct output from find_invalid_longitudes.

        In this case, longitudes in the western hemisphere must be negative.
        """

        these_invalid_indices = geodetic_utils.find_invalid_longitudes(
            longitudes_deg=MESSY_LONGITUDES_DEG,
            sign_in_western_hemisphere=geodetic_utils.NEGATIVE_LONGITUDE_ARG)
        self.assertTrue(numpy.array_equal(
            these_invalid_indices, INVALID_LNG_INDICES_NEGATIVE_IN_WEST))

    def test_get_latlng_centroid(self):
        """Ensures correct output from get_latlng_centroid."""

        (this_centroid_lat_deg, this_centroid_lng_deg
        ) = geodetic_utils.get_latlng_centroid(
            latitudes_deg=POINT_LATITUDES_DEG,
            longitudes_deg=POINT_LONGITUDES_DEG, allow_nan=True)

        self.assertTrue(numpy.isclose(
            this_centroid_lat_deg, CENTROID_LATITUDE_DEG,
            atol=DEFAULT_TOLERANCE))
        self.assertTrue(numpy.isclose(
            this_centroid_lng_deg, CENTROID_LONGITUDE_DEG,
            atol=DEFAULT_TOLERANCE))

    def test_get_elevations(self):
        """Ensures correct output from get_elevations."""

        these_elevations_m_asl = geodetic_utils.get_elevations(
            latitudes_deg=LATITUDES_DEG, longitudes_deg=LONGITUDES_DEG)

        self.assertTrue(numpy.allclose(
            these_elevations_m_asl, ELEVATIONS_M_ASL, atol=DEFAULT_TOLERANCE))

    def test_start_points_and_displacements_to_endpoints(self):
        """Ensures crrctness of start_points_and_displacements_to_endpoints."""

        (these_end_latitudes_deg, these_end_longitudes_deg
        ) = geodetic_utils.start_points_and_displacements_to_endpoints(
            start_latitudes_deg=START_LATITUDES_DEG,
            start_longitudes_deg=START_LONGITUDES_DEG,
            scalar_displacements_metres=START_TO_END_DISTANCES_METRES,
            geodetic_bearings_deg=START_TO_END_BEARINGS_DEG)

        self.assertTrue(numpy.allclose(
            these_end_latitudes_deg, END_LATITUDES_DEG,
            atol=LATLNG_TOLERANCE_DEG))
        self.assertTrue(numpy.allclose(
            these_end_longitudes_deg, END_LONGITUDES_DEG,
            atol=LATLNG_TOLERANCE_DEG))

    def test_xy_to_scalar_displacements_and_bearings(self):
        """Ensures correctness of xy_to_scalar_displacements_and_bearings."""

        (these_scalar_displacements_metres, these_bearings_deg
        ) = geodetic_utils.xy_to_scalar_displacements_and_bearings(
            x_displacements_metres=X_DISPLACEMENTS_METRES,
            y_displacements_metres=Y_DISPLACEMENTS_METRES)

        self.assertTrue(numpy.allclose(
            these_scalar_displacements_metres, SCALAR_DISPLACEMENTS_METRES,
            atol=DEFAULT_TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_bearings_deg, GEODETIC_BEARINGS_DEG, atol=DEFAULT_TOLERANCE))

    def test_scalar_displacements_and_bearings_to_xy(self):
        """Ensures correctness of scalar_displacements_and_bearings_to_xy."""

        (these_x_displacements_metres, these_y_displacements_metres
        ) = geodetic_utils.scalar_displacements_and_bearings_to_xy(
            scalar_displacements_metres=SCALAR_DISPLACEMENTS_METRES,
            geodetic_bearings_deg=GEODETIC_BEARINGS_DEG)

        self.assertTrue(numpy.allclose(
            these_x_displacements_metres, X_DISPLACEMENTS_METRES,
            atol=DEFAULT_TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_y_displacements_metres, Y_DISPLACEMENTS_METRES,
            atol=DEFAULT_TOLERANCE))

    def test_rotate_displacement_vectors_fwd(self):
        """Ensures correct output from rotate_displacement_vectors.

        In this case, converting "forward" (from unprimed to primed coordinate
        system).
        """

        (these_x_prime_displacements_metres, these_y_prime_displacements_metres
        ) = geodetic_utils.rotate_displacement_vectors(
            x_displacements_metres=X_DISPLACEMENTS_METRES,
            y_displacements_metres=Y_DISPLACEMENTS_METRES,
            ccw_rotation_angle_deg=CCW_ROTATION_ANGLE_DEG)

        self.assertTrue(numpy.allclose(
            these_x_prime_displacements_metres, X_PRIME_DISPLACEMENTS_METRES,
            atol=DEFAULT_TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_y_prime_displacements_metres, Y_PRIME_DISPLACEMENTS_METRES,
            atol=DEFAULT_TOLERANCE))

    def test_rotate_displacement_vectors_bwd(self):
        """Ensures correct output from rotate_displacement_vectors.

        In this case, converting "backwards" (from primed to unprimed coordinate
        system).
        """

        (these_x_displacements_metres, these_y_displacements_metres
        ) = geodetic_utils.rotate_displacement_vectors(
            x_displacements_metres=X_PRIME_DISPLACEMENTS_METRES,
            y_displacements_metres=Y_PRIME_DISPLACEMENTS_METRES,
            ccw_rotation_angle_deg=-CCW_ROTATION_ANGLE_DEG)

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
