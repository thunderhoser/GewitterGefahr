"""Unit tests for raw_wind_io.py."""

import unittest
import numpy
import pandas
from gewittergefahr.gg_io import raw_wind_io

TOLERANCE = 1e-6

# The following constants are used to test _check_data_source.
FAKE_PRIMARY_DATA_SOURCE = 'foo'
FAKE_SECONDARY_DATA_SOURCE = 'bar'

# The following constants are used to test
# _primary_and_secondary_sources_to_table.
PRIMARY_SOURCE_BY_PAIR = [
    raw_wind_io.OK_MESONET_DATA_SOURCE, raw_wind_io.STORM_EVENTS_DATA_SOURCE,
    raw_wind_io.HFMETAR_DATA_SOURCE]
PRIMARY_SOURCE_BY_PAIR += (
    [raw_wind_io.MADIS_DATA_SOURCE] * len(raw_wind_io.SECONDARY_DATA_SOURCES))
SECONDARY_SOURCE_BY_PAIR = [None] * 3 + raw_wind_io.SECONDARY_DATA_SOURCES

PRIMARY_AND_SECONDARY_SOURCE_PAIRS_AS_DICT = {
    raw_wind_io.PRIMARY_SOURCE_COLUMN: PRIMARY_SOURCE_BY_PAIR,
    raw_wind_io.SECONDARY_SOURCE_COLUMN: SECONDARY_SOURCE_BY_PAIR}
PRIMARY_AND_SECONDARY_SOURCE_PAIRS_AS_TABLE = pandas.DataFrame.from_dict(
    PRIMARY_AND_SECONDARY_SOURCE_PAIRS_AS_DICT)

# The following constants are used to test _check_elevations.
ELEVATIONS_M_ASL = numpy.array(
    [-1000., 0., 1000., 5000., 10000., numpy.nan, None], dtype=numpy.float64)
ELEV_INVALID_INDICES = numpy.array([0, 4, 5, 6], dtype=int)

# The following constants are used to test _check_latitudes.
LATITUDES_DEG = numpy.array(
    [-100., -90., 0., 90., 1000., numpy.nan, None], dtype=numpy.float64)
LAT_INVALID_INDICES = numpy.array([0, 4, 5, 6], dtype=int)

# The following constants are used to test _check_longitudes*.
LONGITUDES_DEG = numpy.array(
    [-200., -180., 0., 180., 360., 500., numpy.nan, None], dtype=numpy.float64)
LNG_INVALID_INDICES = numpy.array([0, 5, 6, 7], dtype=int)
LNG_INVALID_INDICES_NEGATIVE_IN_WEST = numpy.array([0, 4, 5, 6, 7], dtype=int)
LNG_INVALID_INDICES_POSITIVE_IN_WEST = numpy.array([0, 1, 5, 6, 7], dtype=int)

# The following constants are used to test check_wind_speeds.
SIGNED_WIND_SPEEDS_M_S01 = numpy.array(
    [-100., -50., -10., 0., 10., 50., 100., numpy.nan, None],
    dtype=numpy.float64)
ABSOLUTE_WIND_SPEEDS_M_S01 = numpy.array(
    [-100., -50., -10., 0., 10., 50., 100., numpy.nan, None],
    dtype=numpy.float64)
SIGNED_SPEED_INVALID_INDICES = numpy.array([0, 6, 7, 8], dtype=int)
ABSOLUTE_SPEED_INVALID_INDICES = numpy.array([0, 1, 2, 6, 7, 8], dtype=int)

# The following constants are used to test _check_wind_directions.
WIND_DIRECTIONS_DEG = numpy.array(
    [-10., 0., 180., 359.99, 360., 5000., numpy.nan, None], dtype=numpy.float64)
DIRECTION_INVALID_INDICES = numpy.array([0, 4, 5, 6, 7], dtype=int)

# The following constants are used to test append_source_to_station_id.
STATION_ID_NO_SOURCE = 'CYEG'
NON_MADIS_PRIMARY_SOURCE = 'ok_mesonet'
STATION_ID_NON_MADIS = 'CYEG_ok-mesonet'
SECONDARY_DATA_SOURCE = 'sao'
STATION_ID_MADIS = 'CYEG_madis_sao'

# The following constants are used to test _remove_duplicate_observations.
THESE_LATITUDES_DEG = numpy.array(
    [51.1, 51.102, 51.104, 51.106, 53.5, 53.501, 53.502, 53.503])
THESE_LONGITUDES_DEG = numpy.array(
    [246.0, 246.001, 246.002, 246.1, 246.5, 246.501, 246.502, 246.6])
THESE_TIMES_UNIX_SEC = numpy.array([0, 0, 1, 0, 2, 2, 2, 2], dtype=int)
THESE_U_WINDS_M_S01 = numpy.array(
    [5., 4.999, 5.001, 5.003, 8., 8.001, 8.002, 7.999])
THESE_V_WINDS_M_S01 = numpy.array(
    [-4., -4.001, -4.002, -3.999, 17., 17., 18., 17.])

WIND_DICT_WITH_DUPLICATES = {raw_wind_io.LATITUDE_COLUMN: THESE_LATITUDES_DEG,
                             raw_wind_io.LONGITUDE_COLUMN: THESE_LONGITUDES_DEG,
                             raw_wind_io.TIME_COLUMN: THESE_TIMES_UNIX_SEC,
                             raw_wind_io.U_WIND_COLUMN: THESE_U_WINDS_M_S01,
                             raw_wind_io.V_WIND_COLUMN: THESE_V_WINDS_M_S01}
WIND_TABLE_WITH_DUPLICATES = pandas.DataFrame.from_dict(
    WIND_DICT_WITH_DUPLICATES)
WITH_TABLE_SANS_DUPLICATES = WIND_TABLE_WITH_DUPLICATES.iloc[[0, 2, 3, 4, 6, 7]]

# The following constants are used to test _get_pathless_processed_file_name.
FILE_START_TIME_UNIX_SEC = 1506999600  # 0300 UTC 3 Oct 2017
FILE_END_TIME_UNIX_SEC = 1507003200  # 0400 UTC 3 Oct 2017

PATHLESS_FILE_NAME_MADIS = (
    'wind-observations_madis_sao_2017-10-03-030000_2017-10-03-040000.csv')
PATHLESS_FILE_NAME_NON_MADIS = (
    'wind-observations_ok-mesonet_2017-10-03-030000_2017-10-03-040000.csv')

# The following constants are used to test find_processed_file.
TOP_DIRECTORY_NAME = 'wind'
PROCESSED_FILE_NAME_MADIS = (
    'wind/madis/sao/201710/wind-observations_madis_sao_2017-10-03-030000_'
    '2017-10-03-040000.csv')
PROCESSED_FILE_NAME_NON_MADIS = (
    'wind/ok_mesonet/201710/wind-observations_ok-mesonet_2017-10-03-030000_'
    '2017-10-03-040000.csv')

# The following constants are used to test find_processed_hourly_files.
PERIOD_START_TIME_UNIX_SEC = 1506993753  # 012233 UTC 3 Oct 2017
PERIOD_END_TIME_UNIX_SEC = 1507002295  # 034455 UTC 3 Oct 2017

PROCESSED_HOURLY_FILE_NAMES_MADIS = [
    'wind/madis/sao/201710/'
    'wind-observations_madis_sao_2017-10-03-010000_2017-10-03-015959.csv',
    'wind/madis/sao/201710/'
    'wind-observations_madis_sao_2017-10-03-020000_2017-10-03-025959.csv',
    'wind/madis/sao/201710/'
    'wind-observations_madis_sao_2017-10-03-030000_2017-10-03-035959.csv']

PROCESSED_HOURLY_FILE_NAMES_NON_MADIS = [
    'wind/ok_mesonet/201710/'
    'wind-observations_ok-mesonet_2017-10-03-010000_2017-10-03-015959.csv',
    'wind/ok_mesonet/201710/'
    'wind-observations_ok-mesonet_2017-10-03-020000_2017-10-03-025959.csv',
    'wind/ok_mesonet/201710/'
    'wind-observations_ok-mesonet_2017-10-03-030000_2017-10-03-035959.csv'
]

# The following constants are used to test get_max_of_sustained_and_gust.
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

# The following constants are used to test speed_and_direction_to_uv and
# uv_to_speed_and_direction.
HALF_SQRT_OF_TWO = numpy.sqrt(2.) / 2
EXPECTED_MAX_U_WINDS_M_S01 = numpy.array(
    [0., -12.5 * HALF_SQRT_OF_TWO, -20., -34. * HALF_SQRT_OF_TWO, 0.,
     6.6 * HALF_SQRT_OF_TWO, 1.7, 40. * HALF_SQRT_OF_TWO])
EXPECTED_MAX_V_WINDS_M_S01 = numpy.array(
    [-5., -12.5 * HALF_SQRT_OF_TWO, 0., 34. * HALF_SQRT_OF_TWO, 0.,
     6.6 * HALF_SQRT_OF_TWO, 0., -40. * HALF_SQRT_OF_TWO])


class RawWindIoTests(unittest.TestCase):
    """Each method is a unit test for raw_wind_io.py."""

    def test_check_data_sources_fake_primary(self):
        """Ensures correct output from check_data_sources.

        In this case, primary data source is fake.
        """

        with self.assertRaises(ValueError):
            raw_wind_io.check_data_sources(
                primary_source=FAKE_PRIMARY_DATA_SOURCE)

    def test_check_data_sources_merged_not_allowed(self):
        """Ensures correct output from check_data_sources.

        In this case, primary data source is "merged", which is not allowed.
        """

        with self.assertRaises(ValueError):
            raw_wind_io.check_data_sources(
                primary_source=raw_wind_io.MERGED_DATA_SOURCE,
                allow_merged=False)

    def test_check_data_sources_merged_allowed(self):
        """Ensures correct output from check_data_sources.

        In this case, primary data source is "merged", which is allowed.
        """

        raw_wind_io.check_data_sources(
            primary_source=raw_wind_io.MERGED_DATA_SOURCE, allow_merged=True)

    def test_check_data_sources_fake_secondary(self):
        """Ensures correct output from check_data_sources.

        In this case, secondary data source is fake.
        """

        with self.assertRaises(ValueError):
            raw_wind_io.check_data_sources(
                primary_source=raw_wind_io.MADIS_DATA_SOURCE,
                secondary_source=FAKE_SECONDARY_DATA_SOURCE)

    def test_check_data_sources_madis(self):
        """Ensures correct output from check_data_sources.

        In this case, primary source is MADIS and secondary source is valid.
        """

        raw_wind_io.check_data_sources(
            primary_source=raw_wind_io.MADIS_DATA_SOURCE,
            secondary_source=SECONDARY_DATA_SOURCE)

    def test_check_data_sources_non_madis(self):
        """Ensures correct output from check_data_sources.

        In this case, primary source is non-MADIS.
        """

        raw_wind_io.check_data_sources(primary_source=NON_MADIS_PRIMARY_SOURCE)

    def test_primary_and_secondary_sources_to_table(self):
        """Ensures correctness of _primary_and_secondary_sources_to_table."""

        this_table = raw_wind_io._primary_and_secondary_sources_to_table()
        self.assertTrue(this_table.equals(
            PRIMARY_AND_SECONDARY_SOURCE_PAIRS_AS_TABLE))

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
        """Ensures correct output from check_wind_speeds.

        In this case wind speeds are signed (either u- or v-component), so they
        can be negative.
        """

        these_invalid_indices = raw_wind_io.check_wind_speeds(
            SIGNED_WIND_SPEEDS_M_S01, one_component=True)
        self.assertTrue(numpy.array_equal(these_invalid_indices,
                                          SIGNED_SPEED_INVALID_INDICES))

    def test_check_wind_speeds_absolute(self):
        """Ensures correct output from check_wind_speeds.

        In this case wind speeds are absolute (vector magnitudes), so they
        cannot be negative.
        """

        these_invalid_indices = raw_wind_io.check_wind_speeds(
            ABSOLUTE_WIND_SPEEDS_M_S01, one_component=False)
        self.assertTrue(numpy.array_equal(these_invalid_indices,
                                          ABSOLUTE_SPEED_INVALID_INDICES))

    def test_check_wind_directions(self):
        """Ensures correct output from _check_wind_directions."""

        these_invalid_indices = raw_wind_io._check_wind_directions(
            WIND_DIRECTIONS_DEG)
        self.assertTrue(numpy.array_equal(these_invalid_indices,
                                          DIRECTION_INVALID_INDICES))

    def test_remove_duplicate_observations(self):
        """Ensures correct output from _remove_duplicate_observations."""

        this_wind_table = raw_wind_io._remove_duplicate_observations(
            WIND_TABLE_WITH_DUPLICATES)
        self.assertTrue(this_wind_table.equals(WITH_TABLE_SANS_DUPLICATES))

    def test_get_pathless_processed_file_name_madis(self):
        """Ensures correct output from _get_pathless_processed_file_name.

        In this case, primary data source is MADIS.
        """

        this_pathless_file_name = raw_wind_io._get_pathless_processed_file_name(
            start_time_unix_sec=FILE_START_TIME_UNIX_SEC,
            end_time_unix_sec=FILE_END_TIME_UNIX_SEC,
            primary_source=raw_wind_io.MADIS_DATA_SOURCE,
            secondary_source=SECONDARY_DATA_SOURCE)
        self.assertTrue(this_pathless_file_name == PATHLESS_FILE_NAME_MADIS)

    def test_get_pathless_processed_file_name_non_madis(self):
        """Ensures correct output from _get_pathless_processed_file_name.

        In this case, primary data source is non-MADIS.
        """

        this_pathless_file_name = raw_wind_io._get_pathless_processed_file_name(
            start_time_unix_sec=FILE_START_TIME_UNIX_SEC,
            end_time_unix_sec=FILE_END_TIME_UNIX_SEC,
            primary_source=NON_MADIS_PRIMARY_SOURCE)
        self.assertTrue(this_pathless_file_name == PATHLESS_FILE_NAME_NON_MADIS)

    def test_append_source_to_station_id_madis(self):
        """Ensures correct output from append_source_to_station_id.

        In this case, primary data source is MADIS.
        """

        this_station_id = raw_wind_io.append_source_to_station_id(
            STATION_ID_NO_SOURCE, primary_source=raw_wind_io.MADIS_DATA_SOURCE,
            secondary_source=SECONDARY_DATA_SOURCE)
        self.assertTrue(this_station_id == STATION_ID_MADIS)

    def test_append_source_to_station_id_non_madis(self):
        """Ensures correct output from append_source_to_station_id.

        In this case, primary data source is non-MADIS.
        """

        this_station_id = raw_wind_io.append_source_to_station_id(
            STATION_ID_NO_SOURCE, primary_source=NON_MADIS_PRIMARY_SOURCE)
        self.assertTrue(this_station_id == STATION_ID_NON_MADIS)

    def test_get_max_of_sustained_and_gust(self):
        """Ensures correct output from get_max_of_sustained_and_gust."""

        these_max_wind_speeds_m_s01, these_max_wind_directions_deg = (
            raw_wind_io.get_max_of_sustained_and_gust(
                WIND_SPEEDS_TO_CONVERT_M_S01, WIND_GUST_SPEEDS_TO_CONVERT_M_S01,
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

        In this case, input directions include NaN.
        """

        these_u_winds_m_s01, these_v_winds_m_s01 = (
            raw_wind_io.speed_and_direction_to_uv(
                MAX_WIND_SPEEDS_M_S01, MAX_WIND_DIRECTIONS_WITH_NAN_DEG))

        self.assertTrue(numpy.allclose(
            these_u_winds_m_s01, EXPECTED_MAX_U_WINDS_M_S01, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_v_winds_m_s01, EXPECTED_MAX_V_WINDS_M_S01, atol=TOLERANCE))

    def test_speed_and_direction_to_uv_without_nan(self):
        """Ensures correct output from speed_and_direction_to_uv.

        In this case, input directions do not include NaN.
        """

        these_u_winds_m_s01, these_v_winds_m_s01 = (
            raw_wind_io.speed_and_direction_to_uv(
                MAX_WIND_SPEEDS_M_S01, MAX_WIND_DIRECTIONS_DEG))

        self.assertTrue(numpy.allclose(
            these_u_winds_m_s01, EXPECTED_MAX_U_WINDS_M_S01, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_v_winds_m_s01, EXPECTED_MAX_V_WINDS_M_S01, atol=TOLERANCE))

    def test_uv_to_speed_and_direction(self):
        """Ensures correct output from uv_to_speed_and_direction."""

        these_wind_speeds_m_s01, these_wind_directions_deg = (
            raw_wind_io.uv_to_speed_and_direction(
                EXPECTED_MAX_U_WINDS_M_S01, EXPECTED_MAX_V_WINDS_M_S01))

        self.assertTrue(numpy.allclose(
            these_wind_speeds_m_s01, MAX_WIND_SPEEDS_M_S01, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_wind_directions_deg, MAX_WIND_DIRECTIONS_DEG, atol=TOLERANCE))

    def test_find_processed_file_madis(self):
        """Ensures correct output from find_processed_file.

        In this case, primary data source is MADIS.
        """

        this_file_name = raw_wind_io.find_processed_file(
            start_time_unix_sec=FILE_START_TIME_UNIX_SEC,
            end_time_unix_sec=FILE_END_TIME_UNIX_SEC,
            primary_source=raw_wind_io.MADIS_DATA_SOURCE,
            secondary_source=SECONDARY_DATA_SOURCE,
            top_directory_name=TOP_DIRECTORY_NAME, raise_error_if_missing=False)

        self.assertTrue(this_file_name == PROCESSED_FILE_NAME_MADIS)

    def test_find_processed_file_non_madis(self):
        """Ensures correct output from find_processed_file.

        In this case, primary data source is non-MADIS.
        """

        this_file_name = raw_wind_io.find_processed_file(
            start_time_unix_sec=FILE_START_TIME_UNIX_SEC,
            end_time_unix_sec=FILE_END_TIME_UNIX_SEC,
            primary_source=NON_MADIS_PRIMARY_SOURCE,
            top_directory_name=TOP_DIRECTORY_NAME, raise_error_if_missing=False)

        self.assertTrue(this_file_name == PROCESSED_FILE_NAME_NON_MADIS)

    def test_find_processed_hourly_files_madis(self):
        """Ensures correct output from find_processed_hourly_files.

        In this case, primary data source is MADIS.
        """

        these_file_names, _ = raw_wind_io.find_processed_hourly_files(
            start_time_unix_sec=PERIOD_START_TIME_UNIX_SEC,
            end_time_unix_sec=PERIOD_END_TIME_UNIX_SEC,
            primary_source=raw_wind_io.MADIS_DATA_SOURCE,
            secondary_source=SECONDARY_DATA_SOURCE,
            top_directory_name=TOP_DIRECTORY_NAME, raise_error_if_missing=False)

        self.assertTrue(these_file_names == PROCESSED_HOURLY_FILE_NAMES_MADIS)

    def test_find_processed_hourly_files_non_madis(self):
        """Ensures correct output from find_processed_hourly_files.

        In this case, primary data source is non-MADIS.
        """

        these_file_names, _ = raw_wind_io.find_processed_hourly_files(
            start_time_unix_sec=PERIOD_START_TIME_UNIX_SEC,
            end_time_unix_sec=PERIOD_END_TIME_UNIX_SEC,
            primary_source=NON_MADIS_PRIMARY_SOURCE,
            top_directory_name=TOP_DIRECTORY_NAME, raise_error_if_missing=False)

        self.assertTrue(
            these_file_names == PROCESSED_HOURLY_FILE_NAMES_NON_MADIS)


if __name__ == '__main__':
    unittest.main()
