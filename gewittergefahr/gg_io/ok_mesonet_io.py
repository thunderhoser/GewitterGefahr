"""IO methods for Oklahoma Mesonet data.

Each raw file is space-delimited and contains data for all stations at one
5-minute time step.  Unfortunately there is no way to script the downloading of
these files, because they are not available to the public.  They were provided
to us by Oklahoma Mesonet employees.
"""

import os
import os.path
import numpy
import pandas
from gewittergefahr.gg_io import raw_wind_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): replace main method with named method.

RAW_FILE_EXTENSION = '.mdf'

MINUTES_TO_SECONDS = 60
TIME_FORMAT_YEAR = '%Y'
TIME_FORMAT_MONTH = '%m'
TIME_FORMAT_DAY_OF_MONTH = '%d'
TIME_FORMAT_DATE = '%Y-%m-%d'
TIME_FORMAT_MINUTE = '%Y%m%d%H%M'

STATION_ID_COLUMN_IN_METADATA = 'stid'
STATION_NAME_COLUMN_ORIG = 'name'
LATITUDE_COLUMN_ORIG = 'nlat'
LONGITUDE_COLUMN_ORIG = 'elon'
ELEVATION_COLUMN_ORIG = 'elev'

ORIG_METADATA_COLUMN_NAMES = [
    STATION_ID_COLUMN_IN_METADATA, STATION_NAME_COLUMN_ORIG,
    LATITUDE_COLUMN_ORIG, LONGITUDE_COLUMN_ORIG, ELEVATION_COLUMN_ORIG]

STATION_ID_COLUMN_IN_WIND_DATA = 'STID'
WIND_SPEED_COLUMN_ORIG = 'WSPD'
WIND_DIR_COLUMN_ORIG = 'WDIR'
WIND_GUST_SPEED_COLUMN_ORIG = 'WMAX'
MINUTES_INTO_DAY_COLUMN_ORIG = 'TIME'

ORIG_WIND_DATA_COLUMN_NAMES = [
    STATION_ID_COLUMN_IN_WIND_DATA, WIND_SPEED_COLUMN_ORIG,
    WIND_DIR_COLUMN_ORIG, WIND_GUST_SPEED_COLUMN_ORIG,
    MINUTES_INTO_DAY_COLUMN_ORIG]

# The following constants are used only in the main method.
ORIG_METAFILE_NAME = '/localdata/ryan.lagerquist/aasswp/geoinfo.csv'
ORIG_WIND_FILE_NAME = '/localdata/ryan.lagerquist/aasswp/201007270555.mdf'

NEW_METAFILE_NAME = (
    '/localdata/ryan.lagerquist/aasswp/ok_mesonet_station_metadata.csv')
NEW_WIND_FILE_NAME = (
    '/localdata/ryan.lagerquist/aasswp/ok_mesonet_winds_2010-07-27-055500.csv')


def _is_file_empty(file_name):
    """Determines whether or not file is empty.

    :param file_name: Path to input file.
    :return: file_empty_flag: Boolean flag.  If True, file is empty.
    """

    return os.stat(file_name).st_size == 0


def _date_string_to_unix_sec(date_string):
    """Converts date from string to Unix format.

    :param date_string: String in format "xx yyyy mm dd HH MM SS", where the xx
        is meaningless.
    :return: unix_date_sec: Date in Unix format.
    """

    words = date_string.split()
    date_string = words[1] + '-' + words[2] + '-' + words[3]
    return time_conversion.string_to_unix_sec(date_string, TIME_FORMAT_DATE)


def _read_date_from_raw_file(text_file_name):
    """Reads date from raw file.

    This file should contain all variables for all stations and one 5-minute
    time step.

    :param text_file_name: Path to input file.
    :return: unix_date_sec: Date (seconds, at beginning of day, since 0000 UTC
        1 Jan 1970).
    """

    num_lines_read = 0

    for this_line in open(text_file_name, 'r').readlines():
        num_lines_read += 1
        if num_lines_read == 2:
            return _date_string_to_unix_sec(this_line)


def _remove_invalid_metadata_rows(station_metadata_table):
    """Removes any row with invalid station metadata.

    :param station_metadata_table: pandas DataFrame created by
        read_station_metadata_from_raw_file.
    :return: station_metadata_table: Same as input, with the following
        exceptions.
        [1] Any row with invalid latitude or longitude is removed.
        [2] Any invalid elevation is changed to NaN.
        [3] All longitudes are positive in western hemisphere.
    """

    return raw_wind_io.remove_invalid_rows(
        station_metadata_table, check_lat_flag=True, check_lng_flag=True,
        check_elevation_flag=True)


def _remove_invalid_wind_rows(wind_table):
    """Removes any row with invalid wind data.

    :param wind_table: pandas DataFrame created by read_winds_from_raw_file.
    :return: wind_table: Same as input, with the following exceptions.
        [1] Any row with invalid wind speed is removed.
        [2] Any invalid wind direction is changed to NaN.
    """

    return raw_wind_io.remove_invalid_rows(
        wind_table, check_speed_flag=True, check_direction_flag=True)


def find_local_raw_file(unix_time_sec=None, top_directory_name=None,
                        raise_error_if_missing=False):
    """Finds raw file on local machine.

    This file should contain all variables for all stations and one 5-minute
    time step.

    :param unix_time_sec: Time in Unix format.
    :param top_directory_name: Top-level directory with raw Oklahoma Mesonet
        files.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will raise an error.
    :return: raw_file_name: File path.  If raise_error_if_missing = False and
        file is missing, this will be the *expected* path.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    raw_file_name = '{0:s}/{1:s}/{2:s}/{3:s}/{4:s}{5:s}'.format(
        top_directory_name,
        time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT_YEAR),
        time_conversion.unix_sec_to_string(
            unix_time_sec, TIME_FORMAT_MONTH).lower(),
        time_conversion.unix_sec_to_string(
            unix_time_sec, TIME_FORMAT_DAY_OF_MONTH),
        time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT_MINUTE),
        RAW_FILE_EXTENSION)

    if raise_error_if_missing and not os.path.isfile(raw_file_name):
        raise ValueError(
            'Cannot find raw file.  Expected at location: ' + raw_file_name)

    return raw_file_name


def read_station_metadata_from_raw_file(csv_file_name):
    """Reads metadata for Oklahoma Mesonet stations from raw file.

    This file is provided by the Oklahoma Mesonet and can be found here:
    www.mesonet.org/index.php/api/siteinfo/from_all_active_with_geo_fields/
    format/csv/

    :param csv_file_name: Path to input file.
    :return: station_metadata_table: pandas DataFrame with the following
        columns.
    station_metadata_table.station_id: String ID for station.
    station_metadata_table.station_name: Verbose name for station.
    station_metadata_table.latitude_deg: Latitude (deg N).
    station_metadata_table.longitude_deg: Longitude (deg E).
    station_metadata_table.elevation_m_asl: Elevation (metres above sea level).
    """

    error_checking.assert_file_exists(csv_file_name)

    station_metadata_table = pandas.read_csv(csv_file_name, header=0, sep=',',
                                             dtype={ELEVATION_COLUMN_ORIG:
                                                        numpy.float64})
    station_metadata_table = station_metadata_table[ORIG_METADATA_COLUMN_NAMES]

    column_dict_old_to_new = {
        STATION_ID_COLUMN_IN_METADATA: raw_wind_io.STATION_ID_COLUMN,
        STATION_NAME_COLUMN_ORIG: raw_wind_io.STATION_NAME_COLUMN,
        LATITUDE_COLUMN_ORIG: raw_wind_io.LATITUDE_COLUMN,
        LONGITUDE_COLUMN_ORIG: raw_wind_io.LONGITUDE_COLUMN,
        ELEVATION_COLUMN_ORIG: raw_wind_io.ELEVATION_COLUMN}

    station_metadata_table.rename(columns=column_dict_old_to_new, inplace=True)
    station_metadata_table = _remove_invalid_metadata_rows(
        station_metadata_table)

    num_stations = len(station_metadata_table.index)
    for i in range(num_stations):
        station_metadata_table[raw_wind_io.STATION_ID_COLUMN].values[i] = (
            raw_wind_io.append_source_to_station_id(
                station_metadata_table[raw_wind_io.STATION_ID_COLUMN].values[i],
                primary_source=raw_wind_io.OK_MESONET_DATA_SOURCE))

    return station_metadata_table


def read_winds_from_raw_file(text_file_name):
    """Reads wind observations from raw file.

    This file should contain all variables for all stations and one 5-minute
    time step.

    :param text_file_name: Path to input file.
    :return: wind_table: pandas DataFrame with the following columns.
    wind_table.station_id: String ID for station.
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

    error_checking.assert_file_exists(text_file_name)
    if _is_file_empty(text_file_name):
        wind_dict = {
            raw_wind_io.STATION_ID_COLUMN: numpy.array([], dtype='str'),
            raw_wind_io.TIME_COLUMN: numpy.array([], dtype=int),
            raw_wind_io.WIND_SPEED_COLUMN: numpy.array([]),
            raw_wind_io.WIND_DIR_COLUMN: numpy.array([]),
            raw_wind_io.WIND_GUST_SPEED_COLUMN: numpy.array([]),
            raw_wind_io.WIND_GUST_DIR_COLUMN: numpy.array([])}

        return pandas.DataFrame.from_dict(wind_dict)

    unix_date_sec = _read_date_from_raw_file(text_file_name)
    wind_table = pandas.read_csv(text_file_name, delim_whitespace=True,
                                 skiprows=[0, 1])
    wind_table = wind_table[ORIG_WIND_DATA_COLUMN_NAMES]

    column_dict_old_to_new = {
        STATION_ID_COLUMN_IN_WIND_DATA: raw_wind_io.STATION_ID_COLUMN,
        WIND_SPEED_COLUMN_ORIG: raw_wind_io.WIND_SPEED_COLUMN,
        WIND_DIR_COLUMN_ORIG: raw_wind_io.WIND_DIR_COLUMN,
        WIND_GUST_SPEED_COLUMN_ORIG: raw_wind_io.WIND_GUST_SPEED_COLUMN,
        MINUTES_INTO_DAY_COLUMN_ORIG: raw_wind_io.TIME_COLUMN}
    wind_table.rename(columns=column_dict_old_to_new, inplace=True)

    wind_table[raw_wind_io.TIME_COLUMN] = unix_date_sec + (
        wind_table[raw_wind_io.TIME_COLUMN].values * MINUTES_TO_SECONDS)

    argument_dict = {
        raw_wind_io.WIND_DIR_COLUMN:
            numpy.array(wind_table[raw_wind_io.WIND_DIR_COLUMN].values,
                        dtype=float)}
    wind_table = wind_table.assign(**argument_dict)

    num_observations = len(wind_table.index)
    wind_gust_directions_deg = numpy.full(num_observations, numpy.nan)
    argument_dict = {raw_wind_io.WIND_GUST_DIR_COLUMN: wind_gust_directions_deg}
    wind_table = wind_table.assign(**argument_dict)

    for i in range(num_observations):
        wind_table[raw_wind_io.STATION_ID_COLUMN].values[i] = (
            raw_wind_io.append_source_to_station_id(
                wind_table[raw_wind_io.STATION_ID_COLUMN].values[i],
                raw_wind_io.OK_MESONET_DATA_SOURCE))

    return _remove_invalid_wind_rows(wind_table)


def merge_winds_and_station_metadata(wind_table, station_metadata_table):
    """Merges wind data with metadata for observing stations.

    :param wind_table: pandas DataFrame created by read_winds_from_raw_file.
    :param station_metadata_table: pandas DataFrame created by
        read_station_metadata_from_raw_file.
    :return: wind_table: Same as input but with additional columns listed below.
    wind_table.station_id: String ID for station.
    wind_table.station_name: Verbose name for station.
    wind_table.latitude_deg: Latitude (deg N).
    wind_table.longitude_deg: Longitude (deg E).
    wind_table.elevation_m_asl: Elevation (metres above sea level).
    """

    return wind_table.merge(station_metadata_table,
                            on=raw_wind_io.STATION_ID_COLUMN, how='inner')


if __name__ == '__main__':
    STATION_METADATA_TABLE = read_station_metadata_from_raw_file(
        ORIG_METAFILE_NAME)
    raw_wind_io.write_station_metadata_to_processed_file(STATION_METADATA_TABLE,
                                                         NEW_METAFILE_NAME)

    WIND_TABLE = read_winds_from_raw_file(ORIG_WIND_FILE_NAME)
    WIND_TABLE = raw_wind_io.sustained_and_gust_to_uv_max(WIND_TABLE)
    WIND_TABLE = merge_winds_and_station_metadata(WIND_TABLE,
                                                  STATION_METADATA_TABLE)
    print WIND_TABLE

    raw_wind_io.write_processed_file(WIND_TABLE, NEW_WIND_FILE_NAME)
