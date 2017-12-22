"""IO methods for MADIS wind data.

--- DEFINITIONS ---

MADIS (the Meteorological Assimilation Data Ingest System) consists of many
secondary datasets (or "data streams").  We use all secondary datasets with
near-surface wind observations, listed below:

- coop stations
- CRN (Climate Reference Network)
- HCN (Historical Climate Network)
- maritime observations
- mesonet observations (from many different mesonets, sadly not including the
  Oklahoma Mesonet)
- hourly METARs (meteorological aerodrome reports)
- NEPP (New England Pilot Project)
- SAO (I have no idea what this stands for; Canadian stations only)
- urbanet stations
"""

import copy
import os
import os.path
import numpy
import pandas
from gewittergefahr.gg_io import downloads
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_io import raw_wind_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): replace main method with named method.

RAW_FILE_EXTENSION = '.gz'
FTP_SERVER_NAME = 'madis-data.ncep.noaa.gov'
TOP_FTP_DIRECTORY_NAME = 'archive'

HTTP_HOST_NAME = 'https://madis-data.ncep.noaa.gov'
TOP_HTTP_DIRECTORY_NAME = (
    'https://madis-data.ncep.noaa.gov/madisResearch/data/archive')

SECONDARY_SOURCES_IN_LDAD = [
    raw_wind_io.MADIS_COOP_DATA_SOURCE, raw_wind_io.MADIS_CRN_DATA_SOURCE,
    raw_wind_io.MADIS_HCN_DATA_SOURCE, raw_wind_io.HFMETAR_DATA_SOURCE,
    raw_wind_io.MADIS_MESONET_DATA_SOURCE, raw_wind_io.MADIS_NEPP_DATA_SOURCE,
    raw_wind_io.MADIS_URBANET_DATA_SOURCE]
SECONDARY_SOURCES_NON_LDAD = [
    raw_wind_io.MADIS_MARITIME_DATA_SOURCE, raw_wind_io.MADIS_METAR_DATA_SOURCE,
    raw_wind_io.MADIS_SAO_DATA_SOURCE]

TIME_FORMAT_YEAR = '%Y'
TIME_FORMAT_MONTH = '%m'
TIME_FORMAT_MONTH_YEAR = '%Y%m'
TIME_FORMAT_DAY_OF_MONTH = '%d'
TIME_FORMAT_HOUR = '%Y%m%d_%H00'

LOW_QUALITY_FLAGS = ['X', 'Q', 'k', 'B']
DEFAULT_QUALITY_FLAG = 'y'

WIND_SPEED_FLAG_COLUMN = 'wind_speed_flag'
WIND_DIR_FLAG_COLUMN = 'wind_direction_flag'
WIND_GUST_SPEED_FLAG_COLUMN = 'wind_gust_speed_flag'
WIND_GUST_DIR_FLAG_COLUMN = 'wind_gust_direction_flag'

STATION_ID_COLUMN_ORIG = 'stationId'
STATION_NAME_COLUMN_ORIG = 'stationName'
LATITUDE_COLUMN_ORIG = 'latitude'
LONGITUDE_COLUMN_ORIG = 'longitude'
ELEVATION_COLUMN_ORIG = 'elevation'
WIND_SPEED_COLUMN_ORIG = 'windSpeed'
WIND_DIR_COLUMN_ORIG = 'windDir'
WIND_GUST_SPEED_COLUMN_ORIG = 'windGust'
WIND_GUST_DIR_COLUMN_ORIG = 'windDirMax'
TIME_COLUMN_ORIG = 'observationTime'
TIME_COLUMN_ORIG_BACKUP = 'timeObs'
WIND_SPEED_FLAG_COLUMN_ORIG = 'windSpeedDD'
WIND_DIR_FLAG_COLUMN_ORIG = 'windDirDD'
WIND_GUST_SPEED_FLAG_COLUMN_ORIG = 'windGustDD'
WIND_GUST_DIR_FLAG_COLUMN_ORIG = 'windDirMaxDD'

COLUMN_NAMES = [raw_wind_io.STATION_ID_COLUMN, raw_wind_io.STATION_NAME_COLUMN,
                raw_wind_io.LATITUDE_COLUMN, raw_wind_io.LONGITUDE_COLUMN,
                raw_wind_io.ELEVATION_COLUMN, raw_wind_io.WIND_SPEED_COLUMN,
                raw_wind_io.WIND_DIR_COLUMN, raw_wind_io.WIND_GUST_SPEED_COLUMN,
                raw_wind_io.WIND_GUST_DIR_COLUMN, raw_wind_io.TIME_COLUMN,
                raw_wind_io.TIME_COLUMN, WIND_SPEED_FLAG_COLUMN,
                WIND_DIR_FLAG_COLUMN, WIND_GUST_SPEED_FLAG_COLUMN,
                WIND_GUST_DIR_FLAG_COLUMN]

COLUMN_NAMES_ORIG = [STATION_ID_COLUMN_ORIG, STATION_NAME_COLUMN_ORIG,
                     LATITUDE_COLUMN_ORIG, LONGITUDE_COLUMN_ORIG,
                     ELEVATION_COLUMN_ORIG, WIND_SPEED_COLUMN_ORIG,
                     WIND_DIR_COLUMN_ORIG, WIND_GUST_SPEED_COLUMN_ORIG,
                     WIND_GUST_DIR_COLUMN_ORIG, TIME_COLUMN_ORIG,
                     TIME_COLUMN_ORIG_BACKUP, WIND_SPEED_FLAG_COLUMN_ORIG,
                     WIND_DIR_FLAG_COLUMN_ORIG,
                     WIND_GUST_SPEED_FLAG_COLUMN_ORIG,
                     WIND_GUST_DIR_FLAG_COLUMN_ORIG]

# The following constants are used only in the main method.
NETCDF_FILE_NAME = (
    '/localdata/ryan.lagerquist/aasswp/madis_mesonet_2011-06-08-07.netcdf')
CSV_FILE_NAME = (
    '/localdata/ryan.lagerquist/aasswp/madis_mesonet_winds_2011-06-08-07.csv')


def _column_name_orig_to_new(column_name_orig):
    """Converts column name from orig (MADIS) to new (GewitterGefahr) format.

    :param column_name_orig: Column name in original format.
    :return: column_name: Column name in new format.
    """

    orig_column_flags = [s == column_name_orig for s in COLUMN_NAMES_ORIG]
    orig_column_index = numpy.where(orig_column_flags)[0][0]
    return COLUMN_NAMES[orig_column_index]


def _char_matrix_to_string_list(char_matrix):
    """Converts character matrix to list of strings.

    M = number of strings
    N = max number of characters per string

    :param char_matrix: M-by-N character matrix.
    :return: strings: length-M list of strings.
    """

    num_strings = char_matrix.shape[0]
    strings = [''] * num_strings
    for i in range(num_strings):
        strings[i] = ''.join(char_matrix[i, :]).strip()

    return strings


def _get_online_file_name(unix_time_sec, secondary_source, protocol):
    """Generates expected file path on FTP or HTTP server.

    :param unix_time_sec: Valid time.
    :param secondary_source: String ID for secondary data source.
    :param protocol: Protocol (either "http" or "ftp").
    :return: online_file_name: Expected file path on FTP or HTTP server.
    """

    pathless_file_name = _get_pathless_raw_file_name(unix_time_sec)

    if secondary_source in SECONDARY_SOURCES_IN_LDAD:
        first_subdir_name = 'LDAD'
        second_subdir_name = 'netCDF'
    else:
        first_subdir_name = 'point'
        second_subdir_name = 'netcdf'

    if protocol == 'ftp':
        top_directory_name = copy.deepcopy(TOP_FTP_DIRECTORY_NAME)
    elif protocol == 'http':
        top_directory_name = copy.deepcopy(TOP_HTTP_DIRECTORY_NAME)

    online_directory_name = '{0:s}/{1:s}/{2:s}/{3:s}/{4:s}/{5:s}/{6:s}'.format(
        top_directory_name,
        time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT_YEAR),
        time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT_MONTH),
        time_conversion.unix_sec_to_string(
            unix_time_sec, TIME_FORMAT_DAY_OF_MONTH),
        first_subdir_name, secondary_source, second_subdir_name)

    return '{0:s}/{1:s}'.format(online_directory_name, pathless_file_name)


def _remove_invalid_wind_rows(wind_table):
    """Removes any row with invalid wind data.

    :param wind_table: pandas DataFrame created by read_winds_from_raw_file.
    :return: wind_table: Same as input, with the following exceptions.
        [1] Any row with invalid wind speed, latitude, or longitude is removed.
        [2] Any invalid wind direction or elevation is changed to NaN.
        [3] All longitudes are positive in western hemisphere.
    """

    return raw_wind_io.remove_invalid_rows(
        wind_table, check_speed_flag=True, check_direction_flag=True,
        check_lat_flag=True, check_lng_flag=True, check_elevation_flag=True)


def _remove_low_quality_data(wind_table):
    """Removes low-quality wind data.

    Low-quality wind speeds and directions will be changed to NaN.  Any row with
    only low-quality wind speeds (both sustained and gust) will be removed.

    :param wind_table: pandas DataFrame created by read_winds_from_raw_file.
    :return: wind_table: Same as input, with the following exceptions.
        [1] Low-quality wind speeds and directions are changed to NaN.
        [2] Any row with only low-quality wind speeds is removed.
        [3] Quality flags (columns) are removed.
    """

    is_low_quality = [f in LOW_QUALITY_FLAGS for f in
                      wind_table[WIND_SPEED_FLAG_COLUMN].values]
    low_quality_indices = numpy.where(is_low_quality)[0]
    wind_table[raw_wind_io.WIND_SPEED_COLUMN].values[
        low_quality_indices] = numpy.nan

    is_low_quality = [f in LOW_QUALITY_FLAGS for f in
                      wind_table[WIND_DIR_FLAG_COLUMN].values]
    low_quality_indices = numpy.where(is_low_quality)[0]
    wind_table[raw_wind_io.WIND_DIR_COLUMN].values[
        low_quality_indices] = numpy.nan

    is_low_quality = [f in LOW_QUALITY_FLAGS for f in
                      wind_table[WIND_GUST_SPEED_FLAG_COLUMN].values]
    low_quality_indices = numpy.where(is_low_quality)[0]
    wind_table[raw_wind_io.WIND_GUST_SPEED_COLUMN].values[
        low_quality_indices] = numpy.nan

    is_low_quality = [f in LOW_QUALITY_FLAGS for f in
                      wind_table[WIND_GUST_DIR_FLAG_COLUMN].values]
    low_quality_indices = numpy.where(is_low_quality)[0]
    wind_table[raw_wind_io.WIND_GUST_DIR_COLUMN].values[
        low_quality_indices] = numpy.nan

    columns_to_drop = [WIND_SPEED_FLAG_COLUMN, WIND_DIR_FLAG_COLUMN,
                       WIND_GUST_SPEED_FLAG_COLUMN, WIND_GUST_DIR_FLAG_COLUMN]
    wind_table.drop(columns_to_drop, axis=1, inplace=True)

    return wind_table.loc[
        wind_table[[raw_wind_io.WIND_SPEED_COLUMN,
                    raw_wind_io.WIND_GUST_SPEED_COLUMN]].notnull().any(
                        axis=1)]


def _get_pathless_raw_file_name(unix_time_sec):
    """Generates pathless name for raw MADIS file.

    :param unix_time_sec: Time in Unix format.
    :return: pathless_raw_file_name: Pathless name for raw MADIS file.
    """

    return '{0:s}{1:s}'.format(
        time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT_HOUR),
        RAW_FILE_EXTENSION)


def find_local_raw_file(unix_time_sec=None, secondary_source=None,
                        top_directory_name=None, raise_error_if_missing=True):
    """Finds raw file on local machine.

    This file should contain all fields for one secondary data source and one
    hour.

    :param unix_time_sec: Valid time.
    :param secondary_source: String ID for secondary data source.
    :param top_directory_name: Name of top-level directory with raw MADIS files.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will raise an error.
    :return: raw_file_name: Path to raw file.  If raise_error_if_missing = False
        and file is missing, this will be the *expected* path.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    raw_wind_io.check_data_sources(
        raw_wind_io.MADIS_DATA_SOURCE, secondary_source)
    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    pathless_file_name = _get_pathless_raw_file_name(unix_time_sec)

    raw_file_name = '{0:s}/{1:s}/{2:s}/{3:s}'.format(
        top_directory_name, secondary_source,
        time_conversion.unix_sec_to_string(
            unix_time_sec, TIME_FORMAT_MONTH_YEAR), pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(raw_file_name):
        raise ValueError(
            'Cannot find raw file.  Expected at location: ' + raw_file_name)

    return raw_file_name


def download_raw_file(
        unix_time_sec, secondary_source, top_local_directory_name, protocol,
        user_name=None, password=None, raise_error_if_fails=True):
    """Downloads raw file from either FTP or HTTP server.

    :param unix_time_sec: Valid time.
    :param secondary_source: String ID for secondary data source.
    :param top_local_directory_name: Name of top-level directory with raw MADIS
        files on local machine.
    :param protocol: Protocol (either "http" or "ftp").
    :param user_name: User name on FTP or HTTP server.  To login anonymously,
        leave this as None.
    :param password: Password on FTP or HTTP server.  To login anonymously,
        leave this as None.
    :param raise_error_if_fails: Boolean flag.  If True and download fails, this
        method will raise an error.
    :return: local_gzip_file_name: Local path to file that was just downloaded.
        If download failed but raise_error_if_fails = False, this will be None.
    :raises: ValueError: if protocol is neither "ftp" nor "http".
    """

    error_checking.assert_is_string(protocol)
    if protocol not in ['ftp', 'http']:
        error_string = (
            'Protocol should be either "ftp" or "http", not "{0:s}"'.format(
                protocol))
        raise ValueError(error_string)

    raw_wind_io.check_data_sources(
        raw_wind_io.MADIS_DATA_SOURCE, secondary_source)
    online_file_name = _get_online_file_name(
        unix_time_sec=unix_time_sec, secondary_source=secondary_source,
        protocol=protocol)

    local_gzip_file_name = find_local_raw_file(
        unix_time_sec=unix_time_sec, secondary_source=secondary_source,
        top_directory_name=top_local_directory_name,
        raise_error_if_missing=False)

    if protocol == 'ftp':
        return downloads.download_file_via_ftp(
            server_name=FTP_SERVER_NAME, user_name=user_name, password=password,
            ftp_file_name=online_file_name,
            local_file_name=local_gzip_file_name,
            raise_error_if_fails=raise_error_if_fails)

    return downloads.download_file_via_http(
        online_file_name=online_file_name, local_file_name=local_gzip_file_name,
        user_name=user_name, password=password, host_name=HTTP_HOST_NAME,
        raise_error_if_fails=raise_error_if_fails)


def read_winds_from_raw_file(netcdf_file_name, secondary_source=None,
                             raise_error_if_fails=True):
    """Reads wind observations from raw file.

    This file should contain all fields for one secondary data source and one
    hour.

    :param netcdf_file_name: Path to input file.
    :param secondary_source: String ID for secondary data source.
    :param raise_error_if_fails: Boolean flag.  If True and the read fails, this
        method will raise an error.  If False and the read fails, this method
        will return None.
    :return: wind_table: If file cannot be opened and raise_error_if_fails =
        False, this is None.  Otherwise, it is a pandas DataFrame with the
        following columns.
    wind_table.station_id: String ID for station.
    wind_table.station_name: Verbose name for station.
    wind_table.latitude_deg: Latitude (deg N).
    wind_table.longitude_deg: Longitude (deg E).
    wind_table.elevation_m_asl: Elevation (metres above sea level).
    wind_table.unix_time_sec: Observation time (seconds since 0000 UTC 1 Jan
        1970).
    wind_table.wind_speed_m_s01: Speed of sustained wind (m/s).
    wind_table.wind_direction_deg: Direction of sustained wind (degrees of
        origin -- i.e., direction that the wind is coming from -- as per
        meteorological convention).
    wind_table.wind_gust_speed_m_s01: Speed of wind gust (m/s).
    wind_table.wind_gust_direction_deg: Direction of wind gust (degrees of
        origin).
    """

    error_checking.assert_file_exists(netcdf_file_name)
    netcdf_dataset = netcdf_io.open_netcdf(netcdf_file_name,
                                           raise_error_if_fails)
    if netcdf_dataset is None:
        return None

    station_names = _char_matrix_to_string_list(
        netcdf_dataset.variables[STATION_NAME_COLUMN_ORIG][:])

    try:
        station_ids = _char_matrix_to_string_list(
            netcdf_dataset.variables[STATION_ID_COLUMN_ORIG][:])
    except KeyError:
        station_ids = station_names

    for i in range(len(station_ids)):
        station_ids[i] = raw_wind_io.append_source_to_station_id(
            station_ids[i], primary_source=raw_wind_io.MADIS_DATA_SOURCE,
            secondary_source=secondary_source)

    try:
        unix_times_sec = netcdf_dataset.variables[TIME_COLUMN_ORIG][:]
    except KeyError:
        unix_times_sec = netcdf_dataset.variables[TIME_COLUMN_ORIG_BACKUP][:]

    wind_speeds_m_s01 = netcdf_dataset.variables[WIND_SPEED_COLUMN_ORIG][:]
    wind_speed_quality_flags = netcdf_dataset.variables[
        WIND_SPEED_FLAG_COLUMN_ORIG][:]
    num_observations = len(wind_speeds_m_s01)

    try:
        wind_directions_deg = netcdf_dataset.variables[WIND_DIR_COLUMN_ORIG][:]
        wind_dir_quality_flags = netcdf_dataset.variables[
            WIND_DIR_FLAG_COLUMN_ORIG][:]
    except KeyError:
        wind_directions_deg = numpy.full(num_observations, numpy.nan)
        wind_dir_quality_flags = [DEFAULT_QUALITY_FLAG] * num_observations

    try:
        wind_gust_speeds_m_s01 = netcdf_dataset.variables[
            WIND_GUST_SPEED_COLUMN_ORIG][:]
        wind_gust_speed_quality_flags = netcdf_dataset.variables[
            WIND_GUST_SPEED_FLAG_COLUMN_ORIG][:]
    except KeyError:
        wind_gust_speeds_m_s01 = numpy.full(num_observations, numpy.nan)
        wind_gust_speed_quality_flags = (
            [DEFAULT_QUALITY_FLAG] * num_observations)

    try:
        wind_gust_directions_deg = netcdf_dataset.variables[
            WIND_GUST_DIR_COLUMN_ORIG][:]
        wind_gust_dir_quality_flags = netcdf_dataset.variables[
            WIND_GUST_DIR_FLAG_COLUMN_ORIG][:]
    except KeyError:
        wind_gust_directions_deg = numpy.full(num_observations, numpy.nan)
        wind_gust_dir_quality_flags = [DEFAULT_QUALITY_FLAG] * num_observations

    wind_dict = {raw_wind_io.STATION_ID_COLUMN: station_ids,
                 raw_wind_io.STATION_NAME_COLUMN: station_names,
                 raw_wind_io.LATITUDE_COLUMN: netcdf_dataset.variables[
                     LATITUDE_COLUMN_ORIG][:],
                 raw_wind_io.LONGITUDE_COLUMN: netcdf_dataset.variables[
                     LONGITUDE_COLUMN_ORIG][:],
                 raw_wind_io.ELEVATION_COLUMN: netcdf_dataset.variables[
                     ELEVATION_COLUMN_ORIG][:],
                 raw_wind_io.TIME_COLUMN: numpy.array(unix_times_sec).astype(
                     int),
                 raw_wind_io.WIND_SPEED_COLUMN: wind_speeds_m_s01,
                 raw_wind_io.WIND_DIR_COLUMN: wind_directions_deg,
                 raw_wind_io.WIND_GUST_SPEED_COLUMN: wind_gust_speeds_m_s01,
                 raw_wind_io.WIND_GUST_DIR_COLUMN: wind_gust_directions_deg,
                 WIND_SPEED_FLAG_COLUMN: wind_speed_quality_flags,
                 WIND_DIR_FLAG_COLUMN: wind_dir_quality_flags,
                 WIND_GUST_SPEED_FLAG_COLUMN: wind_gust_speed_quality_flags,
                 WIND_GUST_DIR_FLAG_COLUMN: wind_gust_dir_quality_flags}

    netcdf_dataset.close()
    wind_table = pandas.DataFrame.from_dict(wind_dict)
    wind_table = _remove_invalid_wind_rows(wind_table)
    return _remove_low_quality_data(wind_table)


if __name__ == '__main__':
    WIND_TABLE = read_winds_from_raw_file(NETCDF_FILE_NAME)
    print WIND_TABLE

    WIND_TABLE = raw_wind_io.sustained_and_gust_to_uv_max(WIND_TABLE)
    print WIND_TABLE

    raw_wind_io.write_processed_file(WIND_TABLE, CSV_FILE_NAME)
