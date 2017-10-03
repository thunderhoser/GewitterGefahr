"""Unit tests for raw_wind_io.py."""

import unittest
import numpy
from gewittergefahr.gg_io import raw_wind_io

TOLERANCE = 1e-6

ELEVATIONS_M_ASL = numpy.array(
    [-1000., 0., 1000., 5000., 10000., numpy.nan, None], dtype=numpy.float64)
ELEV_INVALID_INDICES = numpy.array([0, 4, 5, 6], dtype=int)

LATITUDES_DEG = numpy.array(
    [-100., -90., 0., 90., 1000., numpy.nan, None], dtype=numpy.float64)
LAT_INVALID_INDICES = numpy.array([0, 4, 5, 6], dtype=int)

LONGITUDES_DEG = numpy.array(
    [-200., -180., 0., 180., 360., 500., numpy.nan, None], dtype=numpy.float64)
LNG_INVALID_INDICES = numpy.array([0, 5, 6, 7], dtype=int)
LNG_INVALID_INDICES_NEGATIVE_IN_WEST = numpy.array([0, 4, 5, 6, 7], dtype=int)
LNG_INVALID_INDICES_POSITIVE_IN_WEST = numpy.array([0, 1, 5, 6, 7], dtype=int)

SIGNED_WIND_SPEEDS_M_S01 = numpy.array(
    [-400., -100., -10., 0., 10., 100., 400., numpy.nan, None],
    dtype=numpy.float64)
ABSOLUTE_WIND_SPEEDS_M_S01 = numpy.array(
    [-400., -100., -10., 0., 10., 100., 400., numpy.nan, None],
    dtype=numpy.float64)
SIGNED_SPEED_INVALID_INDICES = numpy.array([0, 6, 7, 8], dtype=int)
ABSOLUTE_SPEED_INVALID_INDICES = numpy.array([0, 1, 2, 6, 7, 8], dtype=int)

WIND_DIRECTIONS_DEG = numpy.array(
    [-10., 0., 180., 359.99, 360., 5000., numpy.nan, None], dtype=numpy.float64)
DIRECTION_INVALID_INDICES = numpy.array([0, 4, 5, 6, 7], dtype=int)

STATION_ID_WITHOUT_SOURCE = 'CYEG'
DATA_SOURCE = 'madis'
STATION_ID_WITH_SOURCE = 'CYEG_madis'

FILE_START_TIME_UNIX_SEC = 1506999600  # 0300 UTC 3 Oct 2017
FILE_END_TIME_UNIX_SEC = 1507003200  # 0400 UTC 3 Oct 2017
MADIS_DATA_SOURCE = 'madis'
MADIS_SUBDATASET = 'nepp'
NON_MADIS_DATA_SOURCE = 'ok_mesonet'

PATHLESS_FILE_NAME_MADIS = (
    'wind-observations_madis_nepp_2017-10-03-030000_2017-10-03-040000.csv')
PATHLESS_FILE_NAME_NON_MADIS = (
    'wind-observations_ok-mesonet_2017-10-03-030000_2017-10-03-040000.csv')

TOP_DIRECTORY_NAME = 'wind'
PROCESSED_FILE_NAME_MADIS = (
    'wind/madis/nepp/201710/wind-observations_madis_nepp_2017-10-03-030000_'
    '2017-10-03-040000.csv')
PROCESSED_FILE_NAME_NON_MADIS = (
    'wind/ok_mesonet/201710/wind-observations_ok-mesonet_2017-10-03-030000_'
    '2017-10-03-040000.csv')

WIND_SPEEDS_TO_CONVERT_M_S01 = numpy.array(
    [5., 10., 20., 30., numpy.nan, 6.6, 0., 40.])
WIND_GUST_SPEEDS_TO_CONVERT_M_S01 = numpy.array(
    [numpy.nan, 12.5, 17.5, 34., 0., numpy.nan, 1.7, 38.])
WIND_DIRECTIONS_TO_CONVERT_DEG = numpy.array(
    [0., 70., 90., 145., 200., 225., 280., 315.])
WIND_GUST_DIRECTIONS_TO_CONVERT_DEG = numpy.array(
    [20., 45., 105., 135., 180., 230.1, 270., 335.])

MAX_WIND_SPEEDS_M_S01 = numpy.array(
    [5., 12.5, 20., 34., 0., 6.6, 1.7, 40.])
MAX_WIND_DIRECTIONS_DEG = numpy.array(
    [0., 45., 90., 135., 180., 225., 270., 315.])
MAX_WIND_DIRECTIONS_WITH_NAN_DEG = numpy.array(
    [numpy.nan, 45., 90., 135., 180., 225., 270., 315.])

HALF_SQRT_OF_TWO = numpy.sqrt(2.) / 2
EXPECTED_MAX_U_WINDS_M_S01 = numpy.array(
    [0., -12.5 * HALF_SQRT_OF_TWO, -20., -34. * HALF_SQRT_OF_TWO, 0.,
     6.6 * HALF_SQRT_OF_TWO, 1.7, 40. * HALF_SQRT_OF_TWO])
EXPECTED_MAX_V_WINDS_M_S01 = numpy.array(
    [-5., -12.5 * HALF_SQRT_OF_TWO, 0., 34. * HALF_SQRT_OF_TWO, 0.,
     6.6 * HALF_SQRT_OF_TWO, 0., -40. * HALF_SQRT_OF_TWO])


class RawWindIoTests(unittest.TestCase):
    """Each method is a unit test for raw_wind_io.py."""

    def test_check_elevations(self):
        """Ensures correct output from _check_elevations."""

        these_invalid_indices = raw_wind_io._check_elevations(ELEVATIONS_M_ASL)
        self.assertTrue(numpy.array_equal(these_invalid_indices,
                                          ELEV_INVALID_INDICES))

    def test_check_latitudes(self):
        """Ensures correct output from _check_latitudes."""

        these_invalid_indices = raw_wind_io._check_latitudes(LATITUDES_DEG)
        self.assertTrue(numpy.array_equal(these_invalid_indices,
                                          LAT_INVALID_INDICES))

    def test_check_longitudes(self):
        """Ensures correct output from _check_longitudes."""

        these_invalid_indices = raw_wind_io._check_longitudes(LONGITUDES_DEG)
        self.assertTrue(numpy.array_equal(these_invalid_indices,
                                          LNG_INVALID_INDICES))

    def test_check_longitudes_negative_in_west(self):
        """Ensures correct output from _check_longitudes_negative_in_west."""

        these_invalid_indices = raw_wind_io._check_longitudes_negative_in_west(
            LONGITUDES_DEG)
        self.assertTrue(numpy.array_equal(these_invalid_indices,
                                          LNG_INVALID_INDICES_NEGATIVE_IN_WEST))

    def test_check_longitudes_positive_in_west(self):
        """Ensures correct output from _check_longitudes_positive_in_west."""

        these_invalid_indices = raw_wind_io._check_longitudes_positive_in_west(
            LONGITUDES_DEG)
        self.assertTrue(numpy.array_equal(these_invalid_indices,
                                          LNG_INVALID_INDICES_POSITIVE_IN_WEST))

    def test_check_wind_speeds_signed(self):
        """Ensures correct output from _check_wind_speeds.

        In this case wind speeds are signed (either u- or v-component), so they
        can be negative.
        """

        these_invalid_indices = raw_wind_io._check_wind_speeds(
            SIGNED_WIND_SPEEDS_M_S01, one_component=True)
        self.assertTrue(numpy.array_equal(these_invalid_indices,
                                          SIGNED_SPEED_INVALID_INDICES))

    def test_check_wind_speeds_absolute(self):
        """Ensures correct output from _check_wind_speeds.

        In this case wind speeds are absolute (vector magnitudes), so they
        cannot be negative.
        """

        these_invalid_indices = raw_wind_io._check_wind_speeds(
            ABSOLUTE_WIND_SPEEDS_M_S01, one_component=False)
        self.assertTrue(numpy.array_equal(these_invalid_indices,
                                          ABSOLUTE_SPEED_INVALID_INDICES))

    def test_check_wind_directions(self):
        """Ensures correct output from _check_wind_directions."""

        these_invalid_indices = raw_wind_io._check_wind_directions(
            WIND_DIRECTIONS_DEG)
        self.assertTrue(numpy.array_equal(these_invalid_indices,
                                          DIRECTION_INVALID_INDICES))

    def test_get_pathless_name_for_processed_wind_file_madis(self):
        """Ensures correctness of _get_pathless_name_for_processed_wind_file.

        In this case, data source is MADIS.
        """

        this_pathless_file_name = (
            raw_wind_io._get_pathless_name_for_processed_wind_file(
                start_time_unix_sec=FILE_START_TIME_UNIX_SEC,
                end_time_unix_sec=FILE_END_TIME_UNIX_SEC,
                data_source=MADIS_DATA_SOURCE,
                madis_subdataset=MADIS_SUBDATASET))
        self.assertTrue(this_pathless_file_name == PATHLESS_FILE_NAME_MADIS)

    def test_get_pathless_name_for_processed_wind_file_non_madis(self):
        """Ensures correctness of _get_pathless_name_for_processed_wind_file.

        In this case, data source is not MADIS.
        """

        this_pathless_file_name = (
            raw_wind_io._get_pathless_name_for_processed_wind_file(
                start_time_unix_sec=FILE_START_TIME_UNIX_SEC,
                end_time_unix_sec=FILE_END_TIME_UNIX_SEC,
                data_source=NON_MADIS_DATA_SOURCE))
        self.assertTrue(this_pathless_file_name == PATHLESS_FILE_NAME_NON_MADIS)

    def test_append_source_to_station_id(self):
        """Ensures correct output from append_source_to_station_id."""

        this_station_id = raw_wind_io.append_source_to_station_id(
            STATION_ID_WITHOUT_SOURCE, DATA_SOURCE)
        self.assertTrue(this_station_id, STATION_ID_WITH_SOURCE)

    def test_get_max_of_sustained_and_gust(self):
        """Ensures correct output from _get_max_of_sustained_and_gust."""

        (these_max_wind_speeds_m_s01,
         these_max_wind_directions_deg) = (
             raw_wind_io.get_max_of_sustained_and_gust(
                 WIND_SPEEDS_TO_CONVERT_M_S01,
                 WIND_GUST_SPEEDS_TO_CONVERT_M_S01,
                 WIND_DIRECTIONS_TO_CONVERT_DEG,
                 WIND_GUST_DIRECTIONS_TO_CONVERT_DEG))

        self.assertTrue(numpy.allclose(
            these_max_wind_speeds_m_s01, MAX_WIND_SPEEDS_M_S01, atol=TOLERANCE,
            equal_nan=True))
        self.assertTrue(numpy.allclose(
            these_max_wind_directions_deg, MAX_WIND_DIRECTIONS_DEG,
            atol=TOLERANCE))

    def test_speed_and_direction_to_uv_with_nan(self):
        """Ensures correct output from speed_and_direction_to_uv.

        In this case, input array of directions contains NaN.
        """

        (these_u_winds_m_s01,
         these_v_winds_m_s01) = raw_wind_io.speed_and_direction_to_uv(
             MAX_WIND_SPEEDS_M_S01, MAX_WIND_DIRECTIONS_WITH_NAN_DEG)

        self.assertTrue(numpy.allclose(
            these_u_winds_m_s01, EXPECTED_MAX_U_WINDS_M_S01, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_v_winds_m_s01, EXPECTED_MAX_V_WINDS_M_S01, atol=TOLERANCE))

    def test_speed_and_direction_to_uv_without_nan(self):
        """Ensures correct output from speed_and_direction_to_uv.

        In this case, input array of directions does not contain NaN.
        """

        (these_u_winds_m_s01,
         these_v_winds_m_s01) = raw_wind_io.speed_and_direction_to_uv(
             MAX_WIND_SPEEDS_M_S01, MAX_WIND_DIRECTIONS_DEG)

        self.assertTrue(numpy.allclose(
            these_u_winds_m_s01, EXPECTED_MAX_U_WINDS_M_S01, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_v_winds_m_s01, EXPECTED_MAX_V_WINDS_M_S01, atol=TOLERANCE))

    def test_uv_to_speed_and_direction(self):
        """Ensures correct output from uv_to_speed_and_direction."""

        (these_wind_speeds_m_s01,
         these_wind_directions_deg) = raw_wind_io.uv_to_speed_and_direction(
             EXPECTED_MAX_U_WINDS_M_S01, EXPECTED_MAX_V_WINDS_M_S01)

        self.assertTrue(numpy.allclose(
            these_wind_speeds_m_s01, MAX_WIND_SPEEDS_M_S01, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_wind_directions_deg, MAX_WIND_DIRECTIONS_DEG, atol=TOLERANCE))

    def test_find_processed_wind_file_madis(self):
        """Ensures correct output from find_processed_wind_file.

        In this case, data source is MADIS.
        """

        this_file_name = raw_wind_io.find_processed_wind_file(
            start_time_unix_sec=FILE_START_TIME_UNIX_SEC,
            end_time_unix_sec=FILE_END_TIME_UNIX_SEC,
            data_source=MADIS_DATA_SOURCE, madis_subdataset=MADIS_SUBDATASET,
            top_directory_name=TOP_DIRECTORY_NAME, raise_error_if_missing=False)
        self.assertTrue(this_file_name == PROCESSED_FILE_NAME_MADIS)

    def test_find_processed_wind_file_non_madis(self):
        """Ensures correct output from find_processed_wind_file.

        In this case, data source is not MADIS.
        """

        this_file_name = raw_wind_io.find_processed_wind_file(
            start_time_unix_sec=FILE_START_TIME_UNIX_SEC,
            end_time_unix_sec=FILE_END_TIME_UNIX_SEC,
            data_source=NON_MADIS_DATA_SOURCE,
            top_directory_name=TOP_DIRECTORY_NAME, raise_error_if_missing=False)
        self.assertTrue(this_file_name == PROCESSED_FILE_NAME_NON_MADIS)


if __name__ == '__main__':
    unittest.main()
