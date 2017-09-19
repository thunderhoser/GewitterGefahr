"""IO methods for Oklahoma Mesonet data."""

import numpy
import pandas
import time
import calendar
from gewittergefahr.gg_io import myrorss_io
from gewittergefahr.gg_io import raw_wind_io

# TODO(thunderhoser): add error-checking to all methods.
# TODO(thunderhoser): replace main method with named high-level method.

ORIG_METAFILE_NAME = '/localdata/ryan.lagerquist/aasswp/geoinfo.csv'
ORIG_WIND_FILE_NAME = '/localdata/ryan.lagerquist/aasswp/201007270555.mdf'

NEW_METAFILE_NAME = (
    '/localdata/ryan.lagerquist/aasswp/ok_mesonet_station_metadata.csv')
NEW_WIND_FILE_NAME = (
    '/localdata/ryan.lagerquist/aasswp/ok_mesonet_winds_2010-07-27-055500.csv')

DATA_SOURCE = 'ok_mesonet'

DATE_FORMAT = '%Y-%m-%d'
MINUTES_TO_SECONDS = 60

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


def _date_string_to_unix_sec(date_string):
    """Converts date from string to Unix format (sec since 0000 UTC 1 Jan 1970).

    :param date_string: Date string (format "xx yyyy mm dd HH MM SS", where the
        xx is meaningless).
    :return: unix_date_sec: Valid date (seconds, at beginning of day, since 0000
        UTC 1 Jan 1970).
    """

    words = date_string.split()
    date_string = words[1] + '-' + words[2] + '-' + words[3]
    return calendar.timegm(time.strptime(date_string, DATE_FORMAT))


def _read_valid_date_from_text(text_file_name):
    """Reads valid date from text file provided by Oklahoma Mesonet.

    :param text_file_name: Path to input file.
    :return: unix_date_sec: Valid date (seconds, at beginning of day, since 0000
        UTC 1 Jan 1970).
    """

    num_lines_read = 0

    for this_line in open(text_file_name, 'r').readlines():
        num_lines_read += 1
        if num_lines_read == 2:
            return _date_string_to_unix_sec(this_line)


def _remove_invalid_station_metadata(station_metadata_table):
    """Removes invalid metadata for Oklahoma Mesonet stations.

    :param station_metadata_table: pandas DataFrame created by
        read_station_metadata_from_orig_csv.
    :return: station_metadata_table: Same as input, except that [a] any row with
        an invalid latitude or longitude has been removed and [b] any invalid
        elevation has been changed to NaN.
    """

    invalid_indices = raw_wind_io.check_latitudes(
        station_metadata_table[raw_wind_io.LATITUDE_COLUMN].values)
    station_metadata_table.drop(station_metadata_table.index[invalid_indices],
                                axis=0, inplace=True)

    invalid_indices = raw_wind_io.check_longitudes(
        station_metadata_table[raw_wind_io.LONGITUDE_COLUMN].values)
    station_metadata_table.drop(station_metadata_table.index[invalid_indices],
                                axis=0, inplace=True)

    invalid_indices = raw_wind_io.check_elevations(
        station_metadata_table[raw_wind_io.ELEVATION_COLUMN].values)
    station_metadata_table[raw_wind_io.ELEVATION_COLUMN].values[
        invalid_indices] = numpy.nan

    station_metadata_table[
        raw_wind_io.LONGITUDE_COLUMN] = myrorss_io.convert_lng_positive_in_west(
        station_metadata_table[raw_wind_io.LONGITUDE_COLUMN].values)
    return station_metadata_table


def _remove_invalid_wind_data(wind_table):
    """Removes any row with invalid wind data.

    "Invalid wind data" means that sustained and/or gust speed is out of range.
    If either wind direction (sustained or gust) is out of range, it will be
    replaced with 0 deg (due north), since we don't really care about direction.

    :param wind_table: pandas DataFrame created by read_winds_from_text.
    :return: wind_table: Same as input, except that [1] invalid rows have been
        removed and [2] invalid wind directions have been changed to 0 deg.
    """

    invalid_sustained_indices = raw_wind_io.check_wind_speeds(
        wind_table[raw_wind_io.WIND_SPEED_COLUMN].values)
    wind_table[raw_wind_io.WIND_SPEED_COLUMN].values[
        invalid_sustained_indices] = numpy.nan

    invalid_gust_indices = raw_wind_io.check_wind_speeds(
        wind_table[raw_wind_io.WIND_GUST_SPEED_COLUMN].values)
    wind_table[raw_wind_io.WIND_GUST_SPEED_COLUMN].values[
        invalid_gust_indices] = numpy.nan

    invalid_indices = list(
        set(invalid_gust_indices).intersection(invalid_sustained_indices))
    wind_table.drop(wind_table.index[invalid_indices], axis=0, inplace=True)

    invalid_indices = raw_wind_io.check_wind_directions(
        wind_table[raw_wind_io.WIND_DIR_COLUMN].values)
    wind_table[raw_wind_io.WIND_DIR_COLUMN].values[
        invalid_indices] = raw_wind_io.WIND_DIR_DEFAULT_DEG

    invalid_indices = raw_wind_io.check_wind_directions(
        wind_table[raw_wind_io.WIND_GUST_DIR_COLUMN].values)
    wind_table[raw_wind_io.WIND_GUST_DIR_COLUMN].values[
        invalid_indices] = raw_wind_io.WIND_DIR_DEFAULT_DEG

    return wind_table


def read_station_metadata_from_orig_csv(csv_file_name):
    """Reads metadata for Oklahoma Mesonet stations from CSV file.

    This CSV file is provided by the Oklahoma Mesonet and can be found here:
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
    return _remove_invalid_station_metadata(station_metadata_table)


def read_winds_from_text(text_file_name):
    """Reads wind data from text file provided by Oklahoma Mesonet.

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

    unix_date_sec = _read_valid_date_from_text(text_file_name)

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

    num_observations = len(wind_table.index)
    wind_gust_directions_deg = numpy.full(num_observations, numpy.nan)
    argument_dict = {raw_wind_io.WIND_GUST_DIR_COLUMN: wind_gust_directions_deg}
    wind_table = wind_table.assign(**argument_dict)

    return _remove_invalid_wind_data(wind_table)


def merge_winds_and_metadata(wind_table, station_metadata_table):
    """Merges wind data with metadata for observing stations.

    :param wind_table: pandas DataFrame created by read_winds_from_text.
    :param station_metadata_table: pandas DataFrame created by
        read_station_metadata_from_orig_csv.
    :return: wind_table: Same as input but with additional columns listed below.
    wind_table.station_id: String ID for station.
    wind_table.station_name: Verbose name for station.
    wind_table.latitude_deg: Latitude (deg N).
    wind_table.longitude_deg: Longitude (deg E).
    wind_table.elevation_m_asl: Elevation (metres above sea level).
    """

    wind_table = wind_table.merge(station_metadata_table,
                                  on=raw_wind_io.STATION_ID_COLUMN, how='inner')

    num_observations = len(wind_table.index)
    for i in range(num_observations):
        wind_table[raw_wind_io.STATION_ID_COLUMN].values[
            i] = raw_wind_io.append_source_to_station_id(
            wind_table[raw_wind_io.STATION_ID_COLUMN].values[i], DATA_SOURCE)

    return wind_table


if __name__ == '__main__':
    station_metadata_table = read_station_metadata_from_orig_csv(
        ORIG_METAFILE_NAME)
    raw_wind_io.write_station_metadata_to_csv(station_metadata_table,
                                              NEW_METAFILE_NAME)

    wind_table = read_winds_from_text(ORIG_WIND_FILE_NAME)
    wind_table = raw_wind_io.sustained_and_gust_to_uv_max(wind_table)
    wind_table = merge_winds_and_metadata(wind_table, station_metadata_table)
    print wind_table

    raw_wind_io.write_winds_to_csv(wind_table, NEW_WIND_FILE_NAME)
