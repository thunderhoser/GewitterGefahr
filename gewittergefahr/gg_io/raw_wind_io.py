"""IO methods for raw wind data, which may come from 4 different sources:

- HFMETARs (high-frequency [1- and 5-minute] meteorological aerodrome reports)
- MADIS (Meteorological Assimilation Data Ingest System)
- Oklahoma Mesonet stations
- Storm Events (dataset with local storm reports)
"""

import copy
import os.path
import numpy
import pandas
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6
WIND_DIR_DEFAULT_DEG = 0.

DEGREES_TO_RADIANS = numpy.pi / 180
RADIANS_TO_DEGREES = 180. / numpy.pi
HOURS_TO_SECONDS = 3600
KT_TO_METRES_PER_SECOND = 1.852 / 3.6

TIME_FORMAT_MONTH_YEAR = '%Y%m'
TIME_FORMAT_SECOND = '%Y-%m-%d-%H%M%S'
PROCESSED_FILE_PREFIX = 'wind-observations'
PROCESSED_FILE_EXTENSION = '.csv'

HFMETAR_DATA_SOURCE = 'hfmetar'
MADIS_DATA_SOURCE = 'madis'
OK_MESONET_DATA_SOURCE = 'ok_mesonet'
STORM_EVENTS_DATA_SOURCE = 'storm_events'
MERGED_DATA_SOURCE = 'merged'

PRIMARY_UNMERGED_DATA_SOURCES = [
    HFMETAR_DATA_SOURCE, MADIS_DATA_SOURCE, OK_MESONET_DATA_SOURCE,
    STORM_EVENTS_DATA_SOURCE]
PRIMARY_DATA_SOURCES = PRIMARY_UNMERGED_DATA_SOURCES + [MERGED_DATA_SOURCE]

MADIS_COOP_DATA_SOURCE = 'coop'
MADIS_CRN_DATA_SOURCE = 'crn'
MADIS_HCN_DATA_SOURCE = 'hcn'
MADIS_HFMETAR_DATA_SOURCE = 'hfmetar'
MADIS_MARITIME_DATA_SOURCE = 'maritime'
MADIS_MESONET_DATA_SOURCE = 'mesonet'
MADIS_METAR_DATA_SOURCE = 'metar'
MADIS_NEPP_DATA_SOURCE = 'nepp'
MADIS_SAO_DATA_SOURCE = 'sao'
MADIS_URBANET_DATA_SOURCE = 'urbanet'

SECONDARY_DATA_SOURCES = [
    MADIS_COOP_DATA_SOURCE, MADIS_CRN_DATA_SOURCE, MADIS_HCN_DATA_SOURCE,
    MADIS_HFMETAR_DATA_SOURCE, MADIS_MARITIME_DATA_SOURCE,
    MADIS_MESONET_DATA_SOURCE, MADIS_METAR_DATA_SOURCE, MADIS_NEPP_DATA_SOURCE,
    MADIS_SAO_DATA_SOURCE, MADIS_URBANET_DATA_SOURCE]

PRIMARY_SOURCE_COLUMN = 'primary_source'
SECONDARY_SOURCE_COLUMN = 'secondary_source'

MIN_WIND_DIRECTION_DEG = 0.
MAX_WIND_DIRECTION_DEG = 360. - TOLERANCE
MIN_SIGNED_WIND_SPEED_M_S01 = -100. * KT_TO_METRES_PER_SECOND
MIN_ABSOLUTE_WIND_SPEED_M_S01 = 0.
MAX_WIND_SPEED_M_S01 = 100. * KT_TO_METRES_PER_SECOND
MIN_ELEVATION_M_ASL = -418.  # Lowest point on land (shore of Dead Sea).
MAX_ELEVATION_M_ASL = 8848.  # Highest point on land (Mount Everest).
MIN_LATITUDE_DEG = -90.
MAX_LATITUDE_DEG = 90.
MIN_LONGITUDE_DEG = -180.
MAX_LONGITUDE_DEG = 360.
MIN_LNG_NEGATIVE_IN_WEST_DEG = -180.
MAX_LNG_NEGATIVE_IN_WEST_DEG = 180.
MIN_LNG_POSITIVE_IN_WEST_DEG = 0.
MAX_LNG_POSITIVE_IN_WEST_DEG = 360.

STATION_ID_COLUMN = 'station_id'
STATION_NAME_COLUMN = 'station_name'
LATITUDE_COLUMN = 'latitude_deg'
LONGITUDE_COLUMN = 'longitude_deg'
ELEVATION_COLUMN = 'elevation_m_asl'
UTC_OFFSET_COLUMN = 'utc_offset_hours'
WIND_SPEED_COLUMN = 'wind_speed_m_s01'
WIND_DIR_COLUMN = 'wind_direction_deg'
WIND_GUST_SPEED_COLUMN = 'wind_gust_speed_m_s01'
WIND_GUST_DIR_COLUMN = 'wind_gust_direction_deg'
U_WIND_COLUMN = 'u_wind_m_s01'
V_WIND_COLUMN = 'v_wind_m_s01'
TIME_COLUMN = 'unix_time_sec'

REQUIRED_STATION_METADATA_COLUMNS = [
    STATION_ID_COLUMN, STATION_NAME_COLUMN, LATITUDE_COLUMN, LONGITUDE_COLUMN,
    ELEVATION_COLUMN]

STATION_METADATA_COLUMNS = REQUIRED_STATION_METADATA_COLUMNS + [
    UTC_OFFSET_COLUMN]

STATION_METADATA_COLUMN_TYPE_DICT = {
    STATION_ID_COLUMN: str, STATION_NAME_COLUMN: str,
    LATITUDE_COLUMN: numpy.float64, LONGITUDE_COLUMN: numpy.float64,
    ELEVATION_COLUMN: numpy.float64, UTC_OFFSET_COLUMN: numpy.float64}

WIND_COLUMNS = REQUIRED_STATION_METADATA_COLUMNS + [TIME_COLUMN, U_WIND_COLUMN,
                                                    V_WIND_COLUMN]
WIND_COLUMN_TYPE_DICT = copy.deepcopy(STATION_METADATA_COLUMN_TYPE_DICT)
WIND_COLUMN_TYPE_DICT.update({TIME_COLUMN: numpy.int64,
                              U_WIND_COLUMN: numpy.float64,
                              V_WIND_COLUMN: numpy.float64})


def _primary_and_secondary_sources_to_table():
    """Creates pandas DataFrame with all pairs of primary/secondary data source.

    :return: primary_and_secondary_source_pairs_as_table: pandas DataFrame with
        columns listed below.
    primary_and_secondary_source_pairs_as_table.primary_source: Name of primary
        data source.
    primary_and_secondary_source_pairs_as_table.secondary_source: Name of
        secondary source (None if primary_source != "madis").
    """

    unique_primary_sources = set(PRIMARY_UNMERGED_DATA_SOURCES)
    unique_primary_sources.remove(MADIS_DATA_SOURCE)
    unique_primary_sources = list(unique_primary_sources)

    num_secondary_sources = len(SECONDARY_DATA_SOURCES)
    primary_sources_to_append = [MADIS_DATA_SOURCE] * num_secondary_sources
    secondary_sources_to_prepend = [None] * len(unique_primary_sources)

    primary_source_by_pair = unique_primary_sources + primary_sources_to_append
    secondary_source_by_pair = (
        secondary_sources_to_prepend + SECONDARY_DATA_SOURCES)

    primary_and_secondary_source_pairs_as_dict = {
        PRIMARY_SOURCE_COLUMN: primary_source_by_pair,
        SECONDARY_SOURCE_COLUMN: secondary_source_by_pair}
    return pandas.DataFrame.from_dict(
        primary_and_secondary_source_pairs_as_dict)


def _check_elevations(elevations_m_asl):
    """Finds invalid surface elevations.

    N = number of elevations

    :param elevations_m_asl: length-N numpy array of elevations (metres above
        sea level).
    :return: invalid_indices: 1-D numpy array with indices of invalid surface
        elevations.  For example, if 5th and 12th elevations are invalid, this
        array will contain 4 and 11.
    """

    error_checking.assert_is_real_numpy_array(elevations_m_asl)
    error_checking.assert_is_numpy_array(elevations_m_asl, num_dimensions=1)

    valid_flags = numpy.logical_and(elevations_m_asl >= MIN_ELEVATION_M_ASL,
                                    elevations_m_asl <= MAX_ELEVATION_M_ASL)
    return numpy.where(numpy.invert(valid_flags))[0]


def _check_latitudes(latitudes_deg):
    """Finds invalid latitudes.

    N = number of latitudes.

    :param latitudes_deg: length-N numpy array of latitudes (deg N).
    :return: invalid_indices: 1-D numpy array with indices of invalid latitudes.
    """

    error_checking.assert_is_real_numpy_array(latitudes_deg)
    error_checking.assert_is_numpy_array(latitudes_deg, num_dimensions=1)

    valid_flags = numpy.logical_and(latitudes_deg >= MIN_LATITUDE_DEG,
                                    latitudes_deg <= MAX_LATITUDE_DEG)
    return numpy.where(numpy.invert(valid_flags))[0]


def _check_longitudes(longitudes_deg):
    """Finds invalid longitudes.

    N = number of longitudes.

    :param longitudes_deg: length-N numpy array of longitudes (deg E).
    :return: invalid_indices: 1-D numpy array with indices of invalid
        longitudes.
    """

    error_checking.assert_is_real_numpy_array(longitudes_deg)
    error_checking.assert_is_numpy_array(longitudes_deg, num_dimensions=1)

    valid_flags = numpy.logical_and(longitudes_deg >= MIN_LONGITUDE_DEG,
                                    longitudes_deg <= MAX_LONGITUDE_DEG)
    return numpy.where(numpy.invert(valid_flags))[0]


def _check_longitudes_negative_in_west(longitudes_deg):
    """Finds invalid longitudes.

    N = number of longitudes.

    :param longitudes_deg: length-N numpy array of longitudes (deg E), where
        values in western hemisphere are negative.
    :return: invalid_indices: 1-D numpy array with indices of invalid
        longitudes.
    """

    error_checking.assert_is_real_numpy_array(longitudes_deg)
    error_checking.assert_is_numpy_array(longitudes_deg, num_dimensions=1)

    valid_flags = numpy.logical_and(
        longitudes_deg >= MIN_LNG_NEGATIVE_IN_WEST_DEG,
        longitudes_deg <= MAX_LNG_NEGATIVE_IN_WEST_DEG)
    return numpy.where(numpy.invert(valid_flags))[0]


def _check_longitudes_positive_in_west(longitudes_deg):
    """Finds invalid longitudes.

    N = number of longitudes.

    :param longitudes_deg: length-N numpy array of longitudes (deg E), where
        values in western hemisphere are positive.
    :return: invalid_indices: 1-D numpy array with indices of invalid
        longitudes.
    """

    error_checking.assert_is_real_numpy_array(longitudes_deg)
    error_checking.assert_is_numpy_array(longitudes_deg, num_dimensions=1)

    valid_flags = numpy.logical_and(
        longitudes_deg >= MIN_LNG_POSITIVE_IN_WEST_DEG,
        longitudes_deg <= MAX_LNG_POSITIVE_IN_WEST_DEG)
    return numpy.where(numpy.invert(valid_flags))[0]


def _check_wind_directions(wind_directions_deg):
    """Finds invalid wind directions.

    N = number of observations

    :param wind_directions_deg: length-N numpy array of wind directions (degrees
        of origin).
    :return: invalid_indices: 1-D numpy array with indices of invalid
        directions.
    """

    error_checking.assert_is_real_numpy_array(wind_directions_deg)
    error_checking.assert_is_numpy_array(wind_directions_deg, num_dimensions=1)

    valid_flags = numpy.logical_and(
        wind_directions_deg >= MIN_WIND_DIRECTION_DEG,
        wind_directions_deg <= MAX_WIND_DIRECTION_DEG)
    return numpy.where(numpy.invert(valid_flags))[0]


def _remove_duplicate_observations(wind_table):
    """Removes duplicate wind observations.

    :param wind_table: See documentation for write_processed_file.
    :return: wind_table: Same as input, but maybe with fewer rows.
    """

    wind_speeds_m_s01 = numpy.sqrt(
        wind_table[U_WIND_COLUMN].values ** 2 +
        wind_table[V_WIND_COLUMN].values ** 2)

    num_observations = len(wind_speeds_m_s01)
    observation_strings = [''] * num_observations
    for i in range(num_observations):
        observation_strings[i] = '{0:.2f}_{1:.2f}_{2:d}_{3:.2f}'.format(
            wind_table[LATITUDE_COLUMN].values[i],
            wind_table[LONGITUDE_COLUMN].values[i],
            wind_table[TIME_COLUMN].values[i], wind_speeds_m_s01[i])

    _, unique_indices = numpy.unique(
        numpy.asarray(observation_strings), return_index=True)
    return wind_table.iloc[unique_indices]


def _get_pathless_processed_file_name(start_time_unix_sec=None,
                                      end_time_unix_sec=None,
                                      primary_source=None,
                                      secondary_source=None):
    """Generates pathless name for processed wind file.

    :param start_time_unix_sec: Start time.
    :param end_time_unix_sec: End time.
    :param primary_source: String ID for primary data source.
    :param secondary_source: String ID for secondary data source.
    :return: pathless_processed_file_name: Pathless name for processed wind
        file.
    """

    if primary_source == MADIS_DATA_SOURCE:
        combined_source = '{0:s}_{1:s}'.format(primary_source, secondary_source)
    else:
        combined_source = primary_source.replace('_', '-')

    return '{0:s}_{1:s}_{2:s}_{3:s}{4:s}'.format(
        PROCESSED_FILE_PREFIX, combined_source,
        time_conversion.unix_sec_to_string(
            start_time_unix_sec, TIME_FORMAT_SECOND),
        time_conversion.unix_sec_to_string(
            end_time_unix_sec, TIME_FORMAT_SECOND),
        PROCESSED_FILE_EXTENSION)


def check_wind_speeds(wind_speeds_m_s01, one_component=False):
    """Finds invalid wind speeds.

    N = number of observations.

    :param wind_speeds_m_s01: length-N numpy array of wind speeds (m/s).
    :param one_component: Boolean flag.  If True, wind speeds are only one
        component (either u or v), which means that they can be negative.  If
        False, wind speeds are absolute (vector magnitudes), so they cannot be
        negative.
    :return: invalid_indices: 1-D numpy array with indices of invalid speeds.
    """

    error_checking.assert_is_real_numpy_array(wind_speeds_m_s01)
    error_checking.assert_is_numpy_array(wind_speeds_m_s01, num_dimensions=1)
    error_checking.assert_is_boolean(one_component)

    if one_component:
        this_min_wind_speed_m_s01 = MIN_SIGNED_WIND_SPEED_M_S01
    else:
        this_min_wind_speed_m_s01 = MIN_ABSOLUTE_WIND_SPEED_M_S01

    valid_flags = numpy.logical_and(
        wind_speeds_m_s01 >= this_min_wind_speed_m_s01,
        wind_speeds_m_s01 <= MAX_WIND_SPEED_M_S01)
    return numpy.where(numpy.invert(valid_flags))[0]


def check_data_sources(
        primary_source, secondary_source=None, allow_merged=False):
    """Ensures that data sources are valid.

    :param primary_source: String ID for primary data source (must be in
        PRIMARY_DATA_SOURCES or PRIMARY_UNMERGED_DATA_SOURCES).
    :param secondary_source: String ID for secondary source.  If primary_source
        != "madis", this should be None.  If primary_source == "madis", this
        must be in SECONDARY_DATA_SOURCES.
    :param allow_merged: Boolean flag.  If True, will allow "merged" as primary
        data source.  If False, will not allow "merged".
    :raises: ValueError: if primary_source or secondary_source is invalid.
    """

    if allow_merged:
        valid_primary_sources = copy.deepcopy(PRIMARY_DATA_SOURCES)
    else:
        valid_primary_sources = copy.deepcopy(PRIMARY_UNMERGED_DATA_SOURCES)

    if primary_source not in valid_primary_sources:
        error_string = (
            '\n\n' + str(valid_primary_sources) + '\n\nValid primary sources ' +
            '(listed above) do not include "' + primary_source + '".')
        raise ValueError(error_string)

    if (primary_source == MADIS_DATA_SOURCE and
            secondary_source not in SECONDARY_DATA_SOURCES):
        error_string = (
            '\n\n' + str(SECONDARY_DATA_SOURCES) + '\n\nValid secondary ' +
            'sources (listed above) do not include "' + secondary_source + '".')
        raise ValueError(error_string)


def append_source_to_station_id(station_id, primary_source=None,
                                secondary_source=None):
    """Appends data source to station ID.

    :param station_id: String ID for station.
    :param primary_source: String ID for primary data source.
    :param secondary_source: String ID for secondary data source.
    :return: station_id: Same as input, but with data source appended.
    """

    check_data_sources(primary_source, secondary_source, allow_merged=False)

    if primary_source == MADIS_DATA_SOURCE:
        combined_source = '{0:s}_{1:s}'.format(primary_source, secondary_source)
    else:
        combined_source = primary_source.replace('_', '-')

    return '{0:s}_{1:s}'.format(station_id, combined_source)


def remove_invalid_rows(input_table, check_speed_flag=False,
                        check_direction_flag=False, check_u_wind_flag=False,
                        check_v_wind_flag=False, check_lat_flag=False,
                        check_lng_flag=False, check_elevation_flag=False,
                        check_time_flag=False):
    """Removes any row with invalid data from pandas DataFrame.

    However, this method does not remove rows with invalid wind direction or
    elevation.  It simply sets the wind direction or elevation to NaN, so that
    it will not be mistaken for valid data.  Also, this method converts
    longitudes to positive (180...360 deg E) in western hemisphere.

    :param input_table: pandas DataFrame.
    :param check_speed_flag: Boolean flag.  If True, will check wind speed.
    :param check_direction_flag: Boolean flag.  If True, will check wind
        direction.
    :param check_u_wind_flag: Boolean flag.  If True, will check u-wind.
    :param check_v_wind_flag: Boolean flag.  If True, will check v-wind.
    :param check_lat_flag: Boolean flag.  If True, will check latitude.
    :param check_lng_flag: Boolean flag.  If True, will check longitude.
    :param check_elevation_flag: Boolean flag.  If True, will check elevation.
    :param check_time_flag: Boolean flag.  If True, will check time.
    :return: output_table: Same as input_table, except that some rows may be
        gone.
    """

    error_checking.assert_is_boolean(check_speed_flag)
    error_checking.assert_is_boolean(check_direction_flag)
    error_checking.assert_is_boolean(check_u_wind_flag)
    error_checking.assert_is_boolean(check_v_wind_flag)
    error_checking.assert_is_boolean(check_lat_flag)
    error_checking.assert_is_boolean(check_lng_flag)
    error_checking.assert_is_boolean(check_elevation_flag)
    error_checking.assert_is_boolean(check_time_flag)

    if check_speed_flag:
        invalid_sustained_indices = check_wind_speeds(
            input_table[WIND_SPEED_COLUMN].values, one_component=False)
        input_table[WIND_SPEED_COLUMN].values[
            invalid_sustained_indices] = numpy.nan

        invalid_gust_indices = check_wind_speeds(
            input_table[WIND_GUST_SPEED_COLUMN].values, one_component=False)
        input_table[WIND_GUST_SPEED_COLUMN].values[
            invalid_gust_indices] = numpy.nan

        invalid_indices = list(
            set(invalid_gust_indices).intersection(invalid_sustained_indices))
        input_table.drop(input_table.index[invalid_indices], axis=0,
                         inplace=True)

    if check_direction_flag:
        invalid_indices = _check_wind_directions(
            input_table[WIND_DIR_COLUMN].values)
        input_table[WIND_DIR_COLUMN].values[invalid_indices] = numpy.nan

        invalid_indices = _check_wind_directions(
            input_table[WIND_GUST_DIR_COLUMN].values)
        input_table[WIND_GUST_DIR_COLUMN].values[invalid_indices] = numpy.nan

    if check_u_wind_flag:
        invalid_indices = check_wind_speeds(input_table[U_WIND_COLUMN].values,
                                             one_component=True)
        input_table.drop(input_table.index[invalid_indices], axis=0,
                         inplace=True)

    if check_v_wind_flag:
        invalid_indices = check_wind_speeds(input_table[V_WIND_COLUMN].values,
                                             one_component=True)
        input_table.drop(input_table.index[invalid_indices], axis=0,
                         inplace=True)

    if check_lat_flag:
        invalid_indices = _check_latitudes(input_table[LATITUDE_COLUMN].values)
        input_table.drop(input_table.index[invalid_indices], axis=0,
                         inplace=True)

    if check_lng_flag:
        invalid_indices = _check_longitudes(
            input_table[LONGITUDE_COLUMN].values)
        input_table.drop(input_table.index[invalid_indices], axis=0,
                         inplace=True)

        input_table[LONGITUDE_COLUMN] = (
            lng_conversion.convert_lng_positive_in_west(
                input_table[LONGITUDE_COLUMN].values))

    if check_elevation_flag:
        invalid_indices = _check_elevations(
            input_table[ELEVATION_COLUMN].values)
        input_table[ELEVATION_COLUMN].values[invalid_indices] = numpy.nan

    if check_time_flag:
        invalid_flags = numpy.invert(input_table[TIME_COLUMN].values > 0)
        invalid_indices = numpy.where(invalid_flags)[0]
        input_table.drop(input_table.index[invalid_indices], axis=0,
                         inplace=True)

    return input_table


def get_max_of_sustained_and_gust(wind_speeds_m_s01, wind_gust_speeds_m_s01,
                                  wind_directions_deg,
                                  wind_gust_directions_deg):
    """Converts wind data from 4 variables to 2.

    Original variables:

    - speed of sustained wind
    - direction of sustained wind
    - speed of wind gust
    - direction of wind gust

    New variables:

    - speed of max wind (whichever is higher between sustained and gust)
    - direction of max wind (whichever is higher between sustained and gust)

    "Whichever is higher between sustained and gust" sounds like a stupid
    phrase, because gust speed should always be >= sustained speed.  However,
    due to quality issues, this is not always the case.

    N = number of wind observations.

    :param wind_speeds_m_s01: length-N numpy array of sustained speeds (m/s).
    :param wind_gust_speeds_m_s01: length-N numpy array of gust speeds (m/s).
    :param wind_directions_deg: length-N numpy array of sustained directions
        (degrees of origin).
    :param wind_gust_directions_deg: length-N numpy array of gust directions
        (degrees of origin).
    :return: max_wind_speeds_m_s01: length-N numpy array of max wind speeds
        (m/s).
    :return: max_wind_directions_deg: length-N numpy array with directions of
        max wind (degrees of origin).
    """

    error_checking.assert_is_real_numpy_array(wind_speeds_m_s01)
    error_checking.assert_is_numpy_array(wind_speeds_m_s01, num_dimensions=1)
    num_observations = len(wind_speeds_m_s01)

    error_checking.assert_is_real_numpy_array(wind_gust_speeds_m_s01)
    error_checking.assert_is_numpy_array(
        wind_gust_speeds_m_s01,
        exact_dimensions=numpy.array([num_observations]))

    error_checking.assert_is_real_numpy_array(wind_directions_deg)
    error_checking.assert_is_numpy_array(
        wind_directions_deg,
        exact_dimensions=numpy.array([num_observations]))

    error_checking.assert_is_real_numpy_array(wind_gust_directions_deg)
    error_checking.assert_is_numpy_array(
        wind_gust_directions_deg,
        exact_dimensions=numpy.array([num_observations]))

    wind_speed_matrix_m_s01 = numpy.vstack(
        (wind_speeds_m_s01, wind_gust_speeds_m_s01))
    wind_direction_matrix_deg = numpy.vstack(
        (wind_directions_deg, wind_gust_directions_deg))

    max_wind_speeds_m_s01 = numpy.nanmax(wind_speed_matrix_m_s01,
                                         axis=0).astype(numpy.float64)
    row_indices = numpy.nanargmax(wind_speed_matrix_m_s01, axis=0)

    num_observations = len(max_wind_speeds_m_s01)
    column_indices = numpy.linspace(0, num_observations - 1,
                                    num=num_observations, dtype=int)

    linear_indices = numpy.ravel_multi_index((row_indices, column_indices),
                                             (2, num_observations))
    all_wind_directions_deg = numpy.reshape(wind_direction_matrix_deg,
                                            2 * num_observations)
    max_wind_directions_deg = all_wind_directions_deg[linear_indices]

    nan_flags = numpy.isnan(max_wind_directions_deg)
    nan_indices = numpy.where(nan_flags)[0]
    for i in nan_indices:
        max_wind_directions_deg[i] = numpy.nanmax(
            wind_direction_matrix_deg[:, i])

    return max_wind_speeds_m_s01, max_wind_directions_deg


def speed_and_direction_to_uv(wind_speeds_m_s01, wind_directions_deg):
    """Converts wind vectors from speed and direction to u- and v-components.

    N = number of wind vectors

    :param wind_speeds_m_s01: length-N numpy array of wind speeds (m/s).
    :param wind_directions_deg: length-N numpy array of wind directions (degrees
        of origin).
    :return: u_winds_m_s01: length-N numpy array of u-components (m/s).
    :return: v_winds_m_s01: length-N numpy array of v-components (m/s).
    """

    error_checking.assert_is_real_numpy_array(wind_speeds_m_s01)
    error_checking.assert_is_numpy_array(wind_speeds_m_s01, num_dimensions=1)
    num_observations = len(wind_speeds_m_s01)

    error_checking.assert_is_real_numpy_array(wind_directions_deg)
    error_checking.assert_is_numpy_array(
        wind_directions_deg,
        exact_dimensions=numpy.array([num_observations]))

    these_wind_directions_deg = copy.deepcopy(wind_directions_deg)
    these_wind_directions_deg[
        numpy.isnan(these_wind_directions_deg)] = WIND_DIR_DEFAULT_DEG

    u_winds_m_s01 = -1 * wind_speeds_m_s01 * numpy.sin(
        these_wind_directions_deg * DEGREES_TO_RADIANS)
    v_winds_m_s01 = -1 * wind_speeds_m_s01 * numpy.cos(
        these_wind_directions_deg * DEGREES_TO_RADIANS)
    return u_winds_m_s01, v_winds_m_s01


def uv_to_speed_and_direction(u_winds_m_s01, v_winds_m_s01):
    """Converts wind vectors from u- and v-components to speed and direction.

    N = number of wind vectors

    :param u_winds_m_s01: length-N numpy array of u-components (m/s).
    :param v_winds_m_s01: length-N numpy array of v-components (m/s).
    :return: wind_speeds_m_s01: length-N numpy array of wind speeds (m/s).
    :return: wind_directions_deg: length-N numpy array of wind directions
        (degrees of origin).
    """

    error_checking.assert_is_real_numpy_array(u_winds_m_s01)
    error_checking.assert_is_numpy_array(u_winds_m_s01, num_dimensions=1)
    num_observations = len(u_winds_m_s01)

    error_checking.assert_is_real_numpy_array(v_winds_m_s01)
    error_checking.assert_is_numpy_array(
        v_winds_m_s01, exact_dimensions=numpy.array([num_observations]))

    wind_directions_deg = RADIANS_TO_DEGREES * numpy.arctan2(-u_winds_m_s01,
                                                             -v_winds_m_s01)
    wind_directions_deg = numpy.mod(wind_directions_deg + 360., 360)

    wind_speeds_m_s01 = numpy.sqrt(u_winds_m_s01 ** 2 + v_winds_m_s01 ** 2)
    return wind_speeds_m_s01, wind_directions_deg


def sustained_and_gust_to_uv_max(wind_table):
    """Converts each wind observation from 4 variables to 2.

    Original variables:

    - speed of sustained wind
    - direction of sustained wind
    - speed of wind gust
    - direction of wind gust

    New variables:

    - speed of max wind (whichever is higher between sustained and gust)
    - direction of max wind (whichever is higher between sustained and gust)

    "Whichever is higher between sustained and gust" sounds like a stupid
    phrase, because gust speed should always be >= sustained speed.  However,
    due to quality issues, this is not always the case.

    :param wind_table: pandas DataFrame with at least 4 columns.
    wind_table.wind_speed_m_s01: Speed of sustained wind (m/s).
    wind_table.wind_direction_deg: Direction of sustained wind (degrees of
        origin -- i.e., direction that the wind is coming from -- as per
        meteorological convention).
    wind_table.wind_gust_speed_m_s01: Speed of wind gust (m/s).
    wind_table.wind_gust_direction_deg: Direction of wind gust (degrees of
        origin).
    :return: wind_table: Same as input, but with 4 columns ("wind_speed_m_s01",
        "wind_direction_deg", "wind_gust_speed_m_s01",
        "wind_gust_direction_deg") removed and the following columns added.
    wind_table.u_wind_m_s01: Northward component (m/s) of wind.
    wind_table.v_wind_m_s01: Eastward component (m/s) of wind.
    """

    (max_wind_speeds_m_s01,
     max_wind_directions_deg) = get_max_of_sustained_and_gust(
         wind_table[WIND_SPEED_COLUMN].values,
         wind_table[WIND_GUST_SPEED_COLUMN].values,
         wind_table[WIND_DIR_COLUMN].values,
         wind_table[WIND_GUST_DIR_COLUMN].values)

    columns_to_drop = [WIND_SPEED_COLUMN, WIND_DIR_COLUMN,
                       WIND_GUST_SPEED_COLUMN, WIND_GUST_DIR_COLUMN]
    wind_table.drop(columns_to_drop, axis=1, inplace=True)

    u_winds_m_s01, v_winds_m_s01 = speed_and_direction_to_uv(
        max_wind_speeds_m_s01, max_wind_directions_deg)

    argument_dict = {U_WIND_COLUMN: u_winds_m_s01, V_WIND_COLUMN: v_winds_m_s01}
    return wind_table.assign(**argument_dict)


def write_station_metadata_to_processed_file(station_metadata_table,
                                             csv_file_name):
    """Writes metadata for weather stations to file.

    This is considered a "processed file," as opposed to a "raw file".  A "raw
    file" is one taken directly from another database, in the native format of
    said database.  For examples, see
    `hfmetar_io.read_station_metadata_from_raw_file` and
    `ok_mesonet_io.read_station_metadata_from_raw_file`.

    :param station_metadata_table: pandas DataFrame with the following columns.
    station_metadata_table.station_id: String ID for station.
    station_metadata_table.station_name: Verbose name for station.
    station_metadata_table.latitude_deg: Latitude (deg N).
    station_metadata_table.longitude_deg: Longitude (deg E).
    station_metadata_table.elevation_m_asl: Elevation (metres above sea level).
    station_metadata_table.utc_offset_hours [optional]: Local time minus UTC.
    :param csv_file_name: Path to output file.
    """

    error_checking.assert_columns_in_dataframe(
        station_metadata_table, REQUIRED_STATION_METADATA_COLUMNS)

    file_system_utils.mkdir_recursive_if_necessary(file_name=csv_file_name)
    station_metadata_table.to_csv(
        csv_file_name, header=True, columns=STATION_METADATA_COLUMNS,
        index=False)


def read_station_metadata_from_processed_file(csv_file_name):
    """Reads metadata for weather stations from file.

    :param csv_file_name: Path to input file.
    :return: station_metadata_table: See documentation for
        write_station_metadata_to_processed_file.
    """

    error_checking.assert_file_exists(csv_file_name)
    return pandas.read_csv(
        csv_file_name, header=0, usecols=STATION_METADATA_COLUMNS,
        dtype=STATION_METADATA_COLUMN_TYPE_DICT)


def find_processed_file(start_time_unix_sec=None, end_time_unix_sec=None,
                        primary_source=None, secondary_source=None,
                        top_directory_name=None, raise_error_if_missing=True):
    """Finds processed wind file on local machine.

    :param start_time_unix_sec: Start time.
    :param end_time_unix_sec: End time.
    :param primary_source: String ID for primary data source.
    :param secondary_source: String ID for secondary data source.
    :param top_directory_name: Name of top-level directory with processed wind
        files.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will raise an error.
    :return: processed_file_name: Path to processed wind file.  If
        raise_error_if_missing = False and file is missing, this will be the
        *expected* path.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    check_data_sources(primary_source, secondary_source, allow_merged=True)
    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    pathless_file_name = _get_pathless_processed_file_name(
        start_time_unix_sec=start_time_unix_sec,
        end_time_unix_sec=end_time_unix_sec, primary_source=primary_source,
        secondary_source=secondary_source)

    if primary_source == MADIS_DATA_SOURCE:
        combined_source = '{0:s}/{1:s}'.format(primary_source, secondary_source)
    else:
        combined_source = primary_source

    processed_file_name = '{0:s}/{1:s}/{2:s}/{3:s}'.format(
        top_directory_name, combined_source, time_conversion.unix_sec_to_string(
            start_time_unix_sec, TIME_FORMAT_MONTH_YEAR),
        pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(processed_file_name):
        raise ValueError(
            'Cannot find processed wind file.  Expected at location: ' +
            processed_file_name)

    return processed_file_name


def find_processed_hourly_files(start_time_unix_sec=None,
                                end_time_unix_sec=None, primary_source=None,
                                secondary_source=None, top_directory_name=None,
                                raise_error_if_missing=True):
    """Finds processed hourly wind files on local machine.

    N = number of hours in time period (start_time_unix_sec...end_time_unix_sec)

    :param start_time_unix_sec: Beginning of time period.
    :param end_time_unix_sec: End of time period.
    :param primary_source: String ID for primary data source.
    :param secondary_source: String ID for secondary data source.
    :param top_directory_name: Name of top-level directory with processed wind
        files.
    :param raise_error_if_missing: Boolean flag.  If True and *any* file is
        missing, this method will raise an error.
    :return: processed_file_names: length-N list of paths to processed files.
    :return: hours_unix_sec: length-N numpy array of corresponding hours.
    """

    min_hour_unix_sec = int(rounder.floor_to_nearest(
        start_time_unix_sec, HOURS_TO_SECONDS))
    max_hour_unix_sec = int(rounder.floor_to_nearest(
        end_time_unix_sec, HOURS_TO_SECONDS))

    hours_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=min_hour_unix_sec,
        end_time_unix_sec=max_hour_unix_sec, time_interval_sec=HOURS_TO_SECONDS,
        include_endpoint=True)

    num_hours = len(hours_unix_sec)
    processed_file_names = [''] * num_hours

    for i in range(num_hours):
        processed_file_names[i] = find_processed_file(
            start_time_unix_sec=hours_unix_sec[i],
            end_time_unix_sec=hours_unix_sec[i] + HOURS_TO_SECONDS - 1,
            primary_source=primary_source, secondary_source=secondary_source,
            top_directory_name=top_directory_name,
            raise_error_if_missing=raise_error_if_missing)

    return processed_file_names, hours_unix_sec


def write_processed_file(wind_table, csv_file_name=None, write_mode='w'):
    """Writes wind observations to file.

    This is considered a "processed file," as opposed to a "raw file".  A "raw
    file" is one taken directly from another database, in the native format of
    said database.  For examples, see `madis_io.read_winds_from_raw_file` and
    `ok_mesonet_io.read_winds_from_raw_file`.

    :param wind_table: pandas DataFrame with the following columns.
    wind_table.station_id: String ID for station.
    wind_table.station_name: Verbose name for station.
    wind_table.latitude_deg: Latitude (deg N).
    wind_table.longitude_deg: Longitude (deg E).
    wind_table.elevation_m_asl: Elevation (metres above sea level).
    wind_table.unix_time_sec: Valid time in Unix format.
    wind_table.u_wind_m_s01: u-wind (metres per second).
    wind_table.v_wind_m_s01: v-wind (metres per second).

    :param csv_file_name: Path to output file.
    :param write_mode: Any string accepted by the built-in method `open`.
    """

    error_checking.assert_columns_in_dataframe(wind_table, WIND_COLUMNS)
    file_system_utils.mkdir_recursive_if_necessary(file_name=csv_file_name)

    write_header = not os.path.isfile(csv_file_name) or 'w' in write_mode
    wind_table.to_csv(csv_file_name, header=write_header, columns=WIND_COLUMNS,
                      index=False, mode=write_mode)


def write_processed_hourly_files(wind_table, write_mode='w',
                                 start_time_unix_sec=None,
                                 end_time_unix_sec=None, primary_source=None,
                                 secondary_source=None,
                                 top_directory_name=None):
    """Writes wind observations to hourly files.

    N = number of hours in time period (start_time_unix_sec...end_time_unix_sec)

    :param wind_table: See documentation for write_processed_file.
    :param write_mode: Any string accepted by the built-in method `open`.
    :param start_time_unix_sec: Beginning of time period.
    :param end_time_unix_sec: End of time period.
    :param primary_source: String ID for primary data source.
    :param secondary_source: String ID for secondary data source.
    :param top_directory_name: Name of top-level directory with processed wind
        files.
    :return: processed_file_names: length-N list of paths to processed files.
    """

    processed_file_names, hour_start_times_unix_sec = (
        find_processed_hourly_files(
            start_time_unix_sec=start_time_unix_sec,
            end_time_unix_sec=end_time_unix_sec, primary_source=primary_source,
            secondary_source=secondary_source,
            top_directory_name=top_directory_name,
            raise_error_if_missing=False))

    hour_end_times_unix_sec = hour_start_times_unix_sec + HOURS_TO_SECONDS - 1
    num_hours = len(hour_start_times_unix_sec)

    for i in range(num_hours):
        this_hour_flags = numpy.logical_and(
            wind_table[TIME_COLUMN].values >= hour_start_times_unix_sec[i],
            wind_table[TIME_COLUMN].values <= hour_end_times_unix_sec[i])
        this_hour_indices = numpy.where(this_hour_flags)[0]

        write_processed_file(
            wind_table.iloc[this_hour_indices],
            csv_file_name=processed_file_names[i], write_mode=write_mode)

    return processed_file_names


def merge_data_sources_by_hour(
        start_time_unix_sec, end_time_unix_sec, top_directory_name):
    """For each hour in time period, merges files from all data sources.

    N = number of hours in period (start_time_unix_sec...end_time_unix_sec)

    :param start_time_unix_sec: Beginning of time period.
    :param end_time_unix_sec: End of time period.
    :param top_directory_name: Name of top-level directory with processed wind
        files (both input and output).
    :return: merged_file_names: length-N list of paths to output files.
    """

    primary_and_secondary_source_pairs_as_table = (
        _primary_and_secondary_sources_to_table())
    num_combined_sources = len(
        primary_and_secondary_source_pairs_as_table.index)

    for j in range(num_combined_sources):
        if j == 0:
            these_input_file_names, hours_unix_sec = (
                find_processed_hourly_files(
                    start_time_unix_sec=start_time_unix_sec,
                    end_time_unix_sec=end_time_unix_sec,
                    primary_source=primary_and_secondary_source_pairs_as_table[
                        PRIMARY_SOURCE_COLUMN].values[j],
                    secondary_source=
                    primary_and_secondary_source_pairs_as_table[
                        SECONDARY_SOURCE_COLUMN].values[j],
                    top_directory_name=top_directory_name,
                    raise_error_if_missing=True))

            num_hours = len(hours_unix_sec)
            input_file_name_matrix = numpy.full(
                (num_combined_sources, num_hours), None, dtype=object)
            input_file_name_matrix[j, :] = these_input_file_names

        else:
            input_file_name_matrix[j, :], _ = find_processed_hourly_files(
                start_time_unix_sec=start_time_unix_sec,
                end_time_unix_sec=end_time_unix_sec,
                primary_source=primary_and_secondary_source_pairs_as_table[
                    PRIMARY_SOURCE_COLUMN].values[j],
                secondary_source=primary_and_secondary_source_pairs_as_table[
                    SECONDARY_SOURCE_COLUMN].values[j],
                top_directory_name=top_directory_name,
                raise_error_if_missing=True)

    merged_file_names = [''] * num_hours
    for i in range(num_hours):
        this_list_of_wind_tables = [None] * num_combined_sources

        for j in range(num_combined_sources):
            this_list_of_wind_tables[j] = read_processed_file(
                input_file_name_matrix[j, i])

            if j != 0:
                this_list_of_wind_tables[j], _ = (
                    this_list_of_wind_tables[j].align(
                        this_list_of_wind_tables[0], axis=1))

        this_wind_table = pandas.concat(
            this_list_of_wind_tables, axis=0, ignore_index=True)
        this_wind_table = _remove_duplicate_observations(this_wind_table)

        merged_file_names[i] = find_processed_file(
            start_time_unix_sec=hours_unix_sec[i],
            end_time_unix_sec=hours_unix_sec[i] + HOURS_TO_SECONDS - 1,
            primary_source=MERGED_DATA_SOURCE,
            top_directory_name=top_directory_name, raise_error_if_missing=False)

        print 'Writing wind obs for all data sources to "{0:s}"...'.format(
            merged_file_names[i])
        write_processed_file(
            this_wind_table, merged_file_names[i], write_mode='w')

    return merged_file_names


def read_processed_file(csv_file_name):
    """Reads wind observations from processed file.

    Path to input file.
    :return: wind_table: See documentation for write_processed_file.
    """

    error_checking.assert_file_exists(csv_file_name)
    return pandas.read_csv(
        csv_file_name, header=0, usecols=WIND_COLUMNS,
        dtype=WIND_COLUMN_TYPE_DICT)
