"""Unit tests for raw_wind_io.py."""

import unittest
import numpy
from gewittergefahr.gg_io import raw_wind_io

TOLERANCE = 1e-6

STATION_ID_WITHOUT_SOURCE = 'CYEG'
DATA_SOURCE = 'madis'
EXPECTED_STATION_ID_WITH_SOURCE = 'CYEG_madis'

ELEVATIONS_M_ASL = numpy.array(
    [-1000., 0., 1000., 5000., 10000., numpy.nan, numpy.nan])
EXPECTED_ELEV_INVALID_INDICES = numpy.array([0, 4, 5, 6], dtype=int)

LATITUDES_DEG = numpy.array([-100., -90., 0., 90., 1000., numpy.nan, numpy.nan])
EXPECTED_LAT_INVALID_INDICES = numpy.array([0, 4, 5, 6], dtype=int)

LONGITUDES_DEG = numpy.array(
    [-200., -180., 0., 180., 360., 500., numpy.nan, numpy.nan])
EXPECTED_LNG_INVALID_INDICES = numpy.array([0, 5, 6, 7], dtype=int)
EXPECTED_INVALID_INDICES_NEGATIVE_IN_WEST = numpy.array([0, 4, 5, 6, 7],
                                                        dtype=int)
EXPECTED_INVALID_INDICES_POSITIVE_IN_WEST = numpy.array([0, 1, 5, 6, 7],
                                                        dtype=int)

WIND_SPEEDS_M_S01 = numpy.array(
    [-10., 0., 5., 100., 400., numpy.nan, numpy.nan])
EXPECTED_SPEED_INVALID_INDICES = numpy.array([0, 4, 5, 6], dtype=int)

WIND_DIRECTIONS_DEG = numpy.array(
    [-10., 0., 180., 359.99, 360., 5000., numpy.nan, numpy.nan])
EXPECTED_DIRECTION_INVALID_INDICES = numpy.array([0, 4, 5, 6, 7], dtype=int)

WIND_SPEEDS_TO_CONVERT_M_S01 = numpy.array(
    [10., 20., 30., 5., numpy.nan, numpy.nan, 15.])
WIND_GUST_SPEEDS_TO_CONVERT_M_S01 = numpy.array(
    [numpy.nan, numpy.nan, 40., 7.5, 3., 7., 10.])
WIND_DIRECTIONS_TO_CONVERT_DEG = numpy.array(
    [0., 45., 110., 200., 210., 255., 315.])
WIND_GUST_DIRECTIONS_TO_CONVERT_DEG = numpy.array(
    [345., 60., 90., 180., 225., 270., 300.])

EXPECTED_MAX_WIND_SPEEDS_M_S01 = numpy.array([10., 20., 40., 7.5, 3., 7., 15.])
EXPECTED_MAX_WIND_DIRECTIONS_DEG = numpy.array(
    [0., 45., 90., 180., 225., 270., 315.])

HALF_SQRT_OF_TWO = numpy.sqrt(2.) / 2
EXPECTED_MAX_U_WINDS_M_S01 = numpy.array(
    [0., -20. * HALF_SQRT_OF_TWO, -40., 0., 3. * HALF_SQRT_OF_TWO, 7.,
     15. * HALF_SQRT_OF_TWO])
EXPECTED_MAX_V_WINDS_M_S01 = numpy.array(
    [-10., -20. * HALF_SQRT_OF_TWO, 0., 7.5, 3. * HALF_SQRT_OF_TWO, 0.,
     -15. * HALF_SQRT_OF_TWO])


class RawWindIoTests(unittest.TestCase):
    """Each method is a unit test for raw_wind_io.py."""

    def test_append_source_to_station_id(self):
        """Ensures correct output from append_source_to_station_id."""

        station_id_with_source = raw_wind_io.append_source_to_station_id(
            STATION_ID_WITHOUT_SOURCE, DATA_SOURCE)
        self.assertTrue(station_id_with_source, EXPECTED_STATION_ID_WITH_SOURCE)

    def test_check_elevations(self):
        """Ensures correct output from _check_elevations."""

        invalid_indices = raw_wind_io.check_elevations(ELEVATIONS_M_ASL)
        self.assertTrue(
            numpy.array_equal(invalid_indices, EXPECTED_ELEV_INVALID_INDICES))

    def test_check_latitudes(self):
        """Ensures correct output from _check_latitudes."""

        invalid_indices = raw_wind_io.check_latitudes(LATITUDES_DEG)
        self.assertTrue(
            numpy.array_equal(invalid_indices, EXPECTED_LAT_INVALID_INDICES))

    def test_check_longitudes(self):
        """Ensures correct output from _check_longitudes."""

        invalid_indices = raw_wind_io.check_longitudes(LONGITUDES_DEG)
        self.assertTrue(
            numpy.array_equal(invalid_indices, EXPECTED_LNG_INVALID_INDICES))

    def test_check_longitudes_negative_in_west(self):
        """Ensures correct output from _check_longitudes_negative_in_west."""

        invalid_indices = raw_wind_io.check_longitudes_negative_in_west(
            LONGITUDES_DEG)
        self.assertTrue(
            numpy.array_equal(invalid_indices,
                              EXPECTED_INVALID_INDICES_NEGATIVE_IN_WEST))

    def test_check_longitudes_positive_in_west(self):
        """Ensures correct output from _check_longitudes_positive_in_west."""

        invalid_indices = raw_wind_io.check_longitudes_positive_in_west(
            LONGITUDES_DEG)
        self.assertTrue(
            numpy.array_equal(invalid_indices,
                              EXPECTED_INVALID_INDICES_POSITIVE_IN_WEST))

    def test_check_wind_speeds(self):
        """Ensures correct output from _check_wind_speeds."""

        invalid_indices = raw_wind_io.check_wind_speeds(WIND_SPEEDS_M_S01)
        self.assertTrue(
            numpy.array_equal(invalid_indices, EXPECTED_SPEED_INVALID_INDICES))

    def test_check_wind_directions(self):
        """Ensures correct output from _check_wind_directions."""

        invalid_indices = raw_wind_io.check_wind_directions(WIND_DIRECTIONS_DEG)
        self.assertTrue(numpy.array_equal(invalid_indices,
                                          EXPECTED_DIRECTION_INVALID_INDICES))

    def test_get_max_of_sustained_and_gust(self):
        """Ensures correct output from _get_max_of_sustained_and_gust."""

        (max_wind_speeds_m_s01,
         max_wind_directions_deg) = raw_wind_io.get_max_of_sustained_and_gust(
             WIND_SPEEDS_TO_CONVERT_M_S01, WIND_GUST_SPEEDS_TO_CONVERT_M_S01,
             WIND_DIRECTIONS_TO_CONVERT_DEG,
             WIND_GUST_DIRECTIONS_TO_CONVERT_DEG)

        self.assertTrue(numpy.allclose(max_wind_speeds_m_s01,
                                       EXPECTED_MAX_WIND_SPEEDS_M_S01,
                                       atol=TOLERANCE))
        self.assertTrue(numpy.allclose(max_wind_directions_deg,
                                       EXPECTED_MAX_WIND_DIRECTIONS_DEG,
                                       atol=TOLERANCE))

    def test_speed_and_direction_to_uv(self):
        """Ensures correct output from speed_and_direction_to_uv."""

        (u_winds_m_s01, v_winds_m_s01) = raw_wind_io.speed_and_direction_to_uv(
            EXPECTED_MAX_WIND_SPEEDS_M_S01, EXPECTED_MAX_WIND_DIRECTIONS_DEG)

        self.assertTrue(
            numpy.allclose(u_winds_m_s01, EXPECTED_MAX_U_WINDS_M_S01,
                           atol=TOLERANCE))
        self.assertTrue(
            numpy.allclose(v_winds_m_s01, EXPECTED_MAX_V_WINDS_M_S01,
                           atol=TOLERANCE))

    def test_uv_to_speed_and_direction(self):
        """Ensures correct output from uv_to_speed_and_direction."""

        (wind_speeds_m_s01,
         wind_directions_deg) = raw_wind_io.uv_to_speed_and_direction(
             EXPECTED_MAX_U_WINDS_M_S01, EXPECTED_MAX_V_WINDS_M_S01)

        self.assertTrue(
            numpy.allclose(wind_speeds_m_s01, EXPECTED_MAX_WIND_SPEEDS_M_S01,
                           atol=TOLERANCE))
        self.assertTrue(
            numpy.allclose(wind_directions_deg,
                           EXPECTED_MAX_WIND_DIRECTIONS_DEG,
                           atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
