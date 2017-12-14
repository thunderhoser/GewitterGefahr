"""IO methods for HFMETAR* data.

* HFMETARs = high-frequency (1-minute and 5-minute) meteorological aerodrome
  reports.

HFMETARs are available only for ASOS (Automated Surface Observing System)
stations.
"""

import os.path
import numpy
import pandas
from gewittergefahr.gg_io import downloads
from gewittergefahr.gg_io import raw_wind_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): replace main method with named method.

TIME_FORMAT_YEAR = '%Y'
TIME_FORMAT_MONTH = '%Y%m'
TIME_FORMAT_HOUR_MINUTE = '%Y%m%d%H%M'
RAW_FILE_EXTENSION = '.dat'

PREFIXES_FOR_ONLINE_STATION_ID = ['C', 'K', 'P', 'T']
PATHLESS_FILE_NAME_PREFIX_1MINUTE = '64050'
TOP_ONLINE_DIR_NAME_1MINUTE = 'ftp://ftp.ncdc.noaa.gov/pub/data/asos-onemin'
ONLINE_SUBDIR_PREFIX_1MINUTE = '6405-'

PATHLESS_FILE_NAME_PREFIX_5MINUTE = '64010'
TOP_ONLINE_DIR_NAME_5MINUTE = 'ftp://ftp.ncdc.noaa.gov/pub/data/asos-fivemin'
ONLINE_SUBDIR_PREFIX_5MINUTE = '6401-'

FEET_TO_METRES = 1. / 3.2808
HOURS_TO_SECONDS = 3600
KT_TO_METRES_PER_SECOND = 1.852 / 3.6

STATION_ID_CHAR_INDICES = numpy.array([22, 26], dtype=int)
STATION_NAME_CHAR_INDICES = numpy.array([27, 57], dtype=int)
LATITUDE_CHAR_INDICES = numpy.array([144, 153], dtype=int)
LONGITUDE_CHAR_INDICES = numpy.array([154, 164], dtype=int)
ELEVATION_CHAR_INDICES = numpy.array([165, 171], dtype=int)
UTC_OFFSET_CHAR_INDICES = numpy.array([172, 177], dtype=int)
NUM_HEADER_LINES_IN_METAFILE = 2

LOCAL_DATE_CHAR_INDICES_1MINUTE_FILE = numpy.array([13, 21], dtype=int)
LOCAL_TIME_CHAR_INDICES_1MINUTE_FILE = numpy.array([21, 25], dtype=int)
LOCAL_TIME_CHAR_INDICES_5MINUTE_FILE = numpy.array([13, 25], dtype=int)

METADATA_COLUMNS_TO_MERGE = [
    raw_wind_io.STATION_ID_COLUMN, raw_wind_io.STATION_NAME_COLUMN,
    raw_wind_io.LATITUDE_COLUMN, raw_wind_io.LONGITUDE_COLUMN,
    raw_wind_io.ELEVATION_COLUMN]

# The following constants are used only in the main method.
ORIG_METAFILE_NAME = '/localdata/ryan.lagerquist/aasswp/asos-stations.txt'
NEW_METAFILE_NAME = (
    '/localdata/ryan.lagerquist/aasswp/hfmetar_station_metadata.csv')

ORIG_STATION_ID_1MINUTE = 'KBGD'
STATION_ID_1MINUTE = 'BGD_hfmetar'
MONTH_1MINUTE_UNIX_SEC = 1454284800  # Feb 2016
TOP_LOCAL_DIR_NAME_1MINUTE = '/localdata/ryan.lagerquist/aasswp/hfmetar1'
CSV_FILE_NAME_1MINUTE = (
    '/localdata/ryan.lagerquist/aasswp/hfmetar1/hfmetar1_wind_BGD_201602.csv')

ORIG_STATION_ID_5MINUTE = 'KHKA'
STATION_ID_5MINUTE = 'HKA_hfmetar'
MONTH_5MINUTE_UNIX_SEC = 1410253749  # Sep 2014
TOP_LOCAL_DIR_NAME_5MINUTE = '/localdata/ryan.lagerquist/aasswp/hfmetar5'
CSV_FILE_NAME_5MINUTE = (
    '/localdata/ryan.lagerquist/aasswp/hfmetar5/hfmetar5_wind_HKA_201409.csv')


def _station_id_to_online(station_id):
    """Returns possible versions of station ID used in online database.

    :param station_id: Station ID in GewitterGefahr format (example:
        "YRL_hfmetar").
    :return: possible_online_station_ids: 1-D list of possible station IDs used
        in database.  Example: ["CYRL", "KYRL", "PYRL", "TYRL"].
    """

    start_of_station_id = station_id.split('_')[0]
    possible_online_station_ids = []
    for this_prefix in PREFIXES_FOR_ONLINE_STATION_ID:
        possible_online_station_ids.append('{0:s}{1:s}'.format(
            this_prefix, start_of_station_id))

    return possible_online_station_ids


def _local_time_string_to_unix_sec(local_time_string, utc_offset_hours):
    """Converts time from local string to Unix format.

    :param local_time_string: Local time (format "yyyymmddHHMM").
    :param utc_offset_hours: Local time minus UTC.
    :return: unix_time_sec: Time in Unix format.
    """

    return time_conversion.string_to_unix_sec(
        local_time_string, TIME_FORMAT_HOUR_MINUTE) - (
            utc_offset_hours * HOURS_TO_SECONDS)


def _parse_1minute_wind_from_line(line_string):
    """Parses wind observation from 1-minute-METAR file.

    :param line_string: Line from 1-minute-METAR file.
    :return: wind_speed_kt: Speed of sustained wind (kt).
    wind_direction_deg: Direction of sustained wind (degrees of origin).
    wind_gust_speed_kt: Speed of wind gust (kt).
    wind_gust_direction_deg: Direction of wind gust (degrees of origin).
    """

    words = line_string.split()

    try:
        wind_direction_deg = float(words[-4])
        wind_speed_kt = float(words[-3])
        wind_gust_direction_deg = float(words[-2])
        wind_gust_speed_kt = float(words[-1])
    except (ValueError, IndexError):
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan

    return (wind_speed_kt, wind_direction_deg, wind_gust_speed_kt,
            wind_gust_direction_deg)


def _parse_5minute_wind_from_line(line_string):
    """Parses wind observation from 5-minute-METAR file.

    :param line_string: Line from 5-minute-METAR file.
    :return: wind_speed_kt: Speed of sustained wind (kt).
    wind_direction_deg: Direction of sustained wind (degrees of origin).
    wind_gust_speed_kt: Speed of wind gust (kt).
    wind_gust_direction_deg: Direction of wind gust (degrees of origin).
    """

    wind_gust_direction_deg = numpy.nan
    words = line_string.split()

    try:
        if words[6] == 'AUTO':
            wind_string = words[7]
        else:
            wind_string = words[6]
    except IndexError:
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan

    wind_string = wind_string.upper()
    index_of_kt = wind_string.find('KT')
    if index_of_kt == -1:
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan

    try:
        wind_direction_deg = float(wind_string[0:3])
    except IndexError:
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan
    except ValueError:
        wind_direction_deg = numpy.nan

    index_of_g = wind_string.find('G')

    try:
        if index_of_g == -1:
            if index_of_kt <= 3:
                raise IndexError

            wind_speed_kt = float(wind_string[3:index_of_kt])
        else:
            if index_of_g <= 3:
                raise IndexError

            wind_speed_kt = float(wind_string[3:index_of_g])
    except IndexError:
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan
    except ValueError:
        wind_speed_kt = numpy.nan

    if index_of_g == -1:
        wind_gust_speed_kt = numpy.nan
    else:
        try:
            if index_of_kt <= index_of_g + 1:
                raise IndexError

            wind_gust_speed_kt = float(
                wind_string[(index_of_g + 1):index_of_kt])
        except IndexError:
            return numpy.nan, numpy.nan, numpy.nan, numpy.nan
        except ValueError:
            wind_gust_speed_kt = numpy.nan

    return (wind_speed_kt, wind_direction_deg, wind_gust_speed_kt,
            wind_gust_direction_deg)


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

    :param wind_table: pandas DataFrame created by either
        read_1minute_winds_from_raw_file or read_5minute_winds_from_raw_file.
    :return: wind_table: Same as input, with the following exceptions.
        [1] Any row with invalid wind speed is removed.
        [2] Any invalid wind direction is changed to NaN.
    """

    return raw_wind_io.remove_invalid_rows(
        wind_table, check_speed_flag=True, check_direction_flag=True)


def _get_pathless_raw_1minute_file_name(station_id, month_unix_sec):
    """Generates pathless name for raw 1-minute file.

    :param station_id: String ID for station.
    :param month_unix_sec: Month in Unix format.
    :return: pathless_raw_file_name: Pathless name for raw 1-minute file.
    """

    return '{0:s}{1:s}{2:s}{3:s}'.format(
        PATHLESS_FILE_NAME_PREFIX_1MINUTE, station_id,
        time_conversion.unix_sec_to_string(month_unix_sec, TIME_FORMAT_MONTH),
        RAW_FILE_EXTENSION)


def _get_pathless_raw_5minute_file_name(station_id, month_unix_sec):
    """Generates pathless name for raw 5-minute file.

    :param station_id: String ID for station.
    :param month_unix_sec: Month in Unix format.
    :return: pathless_raw_file_name: Pathless name for raw 5-minute file.
    """

    return '{0:s}{1:s}{2:s}{3:s}'.format(
        PATHLESS_FILE_NAME_PREFIX_5MINUTE, station_id,
        time_conversion.unix_sec_to_string(month_unix_sec, TIME_FORMAT_MONTH),
        RAW_FILE_EXTENSION)


def find_local_raw_1minute_file(station_id=None, month_unix_sec=None,
                                top_directory_name=None,
                                raise_error_if_missing=True):
    """Finds raw 1-minute file on local machine.

    This file should contain 1-minute METARs for one station-month.

    :param station_id: String ID for station.
    :param month_unix_sec: Month in Unix format.
    :param top_directory_name: Top-level directory for raw 1-minute files.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will raise an error.
    :return: raw_1minute_file_name: File path.  If raise_error_if_missing =
        False and file is missing, this will be the *expected* path.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    error_checking.assert_is_string(station_id)
    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    pathless_file_name = _get_pathless_raw_1minute_file_name(station_id,
                                                             month_unix_sec)
    raw_1minute_file_name = '{0:s}/{1:s}/{2:s}'.format(
        top_directory_name, station_id, pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(raw_1minute_file_name):
        raise ValueError(
            'Cannot find raw 1-minute file.  Expected at location: ' +
            raw_1minute_file_name)

    return raw_1minute_file_name


def find_local_raw_5minute_file(station_id=None, month_unix_sec=None,
                                top_directory_name=None,
                                raise_error_if_missing=True):
    """Finds raw 5-minute file on local machine.

    This file should contain 5-minute METARs for one station-month.

    :param station_id: String ID for station.
    :param month_unix_sec: Month in Unix format.
    :param top_directory_name: Top-level directory for raw 1-minute files.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will raise an error.
    :return: raw_1minute_file_name: File path.  If raise_error_if_missing =
        False and file is missing, this will be the *expected* path.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    error_checking.assert_is_string(station_id)
    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    pathless_file_name = _get_pathless_raw_5minute_file_name(station_id,
                                                             month_unix_sec)
    raw_5minute_file_name = '{0:s}/{1:s}/{2:s}'.format(
        top_directory_name, station_id, pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(raw_5minute_file_name):
        raise ValueError(
            'Cannot find raw 5-minute file.  Expected at location: ' +
            raw_5minute_file_name)

    return raw_5minute_file_name


def read_station_metadata_from_raw_file(text_file_name):
    """Reads metadata for ASOS stations from raw file.

    This file is provided by the National Climatic Data Center (NCDC) and can be
    found here: www.ncdc.noaa.gov/homr/file/asos-stations.txt

    :param text_file_name: Path to input file.
    :return: station_metadata_table: pandas DataFrame with the following
        columns.
    station_metadata_table.station_id: String ID for station.
    station_metadata_table.station_name: Verbose name for station.
    station_metadata_table.latitude_deg: Latitude (deg N).
    station_metadata_table.longitude_deg: Longitude (deg E).
    station_metadata_table.elevation_m_asl: Elevation (metres above sea level).
    station_metadata_table.utc_offset_hours: Difference between local station
        time and UTC (local minus UTC).
    """

    error_checking.assert_file_exists(text_file_name)

    num_lines_read = 0
    station_ids = []
    station_names = []
    latitudes_deg = []
    longitudes_deg = []
    elevations_m_asl = []
    utc_offsets_hours = []

    for this_line in open(text_file_name, 'r').readlines():
        num_lines_read += 1
        if num_lines_read <= 2:
            continue

        this_station_id_no_source = (
            this_line[STATION_ID_CHAR_INDICES[0]:STATION_ID_CHAR_INDICES[1]])
        this_station_id = raw_wind_io.append_source_to_station_id(
            this_station_id_no_source.strip(),
            primary_source=raw_wind_io.HFMETAR_DATA_SOURCE)

        this_station_name = (this_line[STATION_NAME_CHAR_INDICES[0]:
                                       STATION_NAME_CHAR_INDICES[1]].strip())

        station_ids.append(this_station_id)
        station_names.append(this_station_name)
        latitudes_deg.append(
            float(this_line[LATITUDE_CHAR_INDICES[0]:LATITUDE_CHAR_INDICES[1]]))
        longitudes_deg.append(float(
            this_line[LONGITUDE_CHAR_INDICES[0]:LONGITUDE_CHAR_INDICES[1]]))
        elevations_m_asl.append(float(
            this_line[ELEVATION_CHAR_INDICES[0]:ELEVATION_CHAR_INDICES[1]]))
        utc_offsets_hours.append(float(
            this_line[UTC_OFFSET_CHAR_INDICES[0]:UTC_OFFSET_CHAR_INDICES[1]]))

    station_metadata_dict = {raw_wind_io.STATION_ID_COLUMN: station_ids,
                             raw_wind_io.STATION_NAME_COLUMN: station_names,
                             raw_wind_io.LATITUDE_COLUMN: latitudes_deg,
                             raw_wind_io.LONGITUDE_COLUMN: longitudes_deg,
                             raw_wind_io.ELEVATION_COLUMN: elevations_m_asl,
                             raw_wind_io.UTC_OFFSET_COLUMN: utc_offsets_hours}

    station_metadata_table = pandas.DataFrame.from_dict(station_metadata_dict)
    station_metadata_table[raw_wind_io.ELEVATION_COLUMN] *= FEET_TO_METRES
    return _remove_invalid_metadata_rows(station_metadata_table)


def download_1minute_file(station_id=None, month_unix_sec=None,
                          top_local_directory_name=None,
                          raise_error_if_fails=True):
    """Downloads file with 1-minute METARs for one station-month.

    :param station_id: String ID for station.
    :param month_unix_sec: Month in Unix format (sec since 0000 UTC 1 Jan 1970).
    :param top_local_directory_name: Top local directory with raw files
        containing 1-minute METARs.
    :param raise_error_if_fails: Boolean flag.  If True and download fails, will
        raise error.
    :return: local_file_name: Path to downloaded file on local machine.  If
        download failed but raise_error_if_fails = False, local_file_name =
        None.
    :raises: ValueError: if file cannot be downloaded.
    """

    local_file_name = find_local_raw_1minute_file(
        station_id=station_id, month_unix_sec=month_unix_sec,
        top_directory_name=top_local_directory_name,
        raise_error_if_missing=False)
    possible_orig_station_ids = _station_id_to_online(station_id)

    for this_station_id in possible_orig_station_ids:
        pathless_file_name = _get_pathless_raw_1minute_file_name(
            this_station_id, month_unix_sec)
        online_file_name = '{0:s}/{1:s}{2:s}/{3:s}'.format(
            TOP_ONLINE_DIR_NAME_1MINUTE, ONLINE_SUBDIR_PREFIX_1MINUTE,
            time_conversion.unix_sec_to_string(
                month_unix_sec, TIME_FORMAT_YEAR),
            pathless_file_name)

        this_local_file_name = downloads.download_file_via_http(
            online_file_name, local_file_name,
            raise_error_if_fails=raise_error_if_fails)

        if this_local_file_name is not None:
            return local_file_name

    if raise_error_if_fails:
        month_string = time_conversion.unix_sec_to_string(
            month_unix_sec, TIME_FORMAT_MONTH)
        raise ValueError(
            'Could not download file for station "{0:s}", month {1:s}.'.format(
                station_id, month_string))

    return None


def download_5minute_file(station_id=None, month_unix_sec=None,
                          top_local_directory_name=None,
                          raise_error_if_fails=True):
    """Downloads file with 5-minute METARs for one station-month.

    :param station_id: String ID for station.
    :param month_unix_sec: Month in Unix format (sec since 0000 UTC 1 Jan 1970).
    :param top_local_directory_name: Top local directory with raw files
        containing 5-minute METARs.
    :param raise_error_if_fails: Boolean flag.  If True and download fails, will
        raise error.
    :return: local_file_name: Path to downloaded file on local machine.  If
        download failed but raise_error_if_fails = False, local_file_name =
        None.
    """

    local_file_name = find_local_raw_5minute_file(
        station_id=station_id, month_unix_sec=month_unix_sec,
        top_directory_name=top_local_directory_name,
        raise_error_if_missing=False)
    possible_orig_station_ids = _station_id_to_online(station_id)

    for this_station_id in possible_orig_station_ids:
        pathless_file_name = _get_pathless_raw_5minute_file_name(
            this_station_id, month_unix_sec)
        online_file_name = '{0:s}/{1:s}{2:s}/{3:s}'.format(
            TOP_ONLINE_DIR_NAME_5MINUTE, ONLINE_SUBDIR_PREFIX_5MINUTE,
            time_conversion.unix_sec_to_string(
                month_unix_sec, TIME_FORMAT_YEAR),
            pathless_file_name)

        this_local_file_name = downloads.download_file_via_http(
            online_file_name, local_file_name,
            raise_error_if_fails=raise_error_if_fails)

        if this_local_file_name is not None:
            return local_file_name

    if raise_error_if_fails:
        month_string = time_conversion.unix_sec_to_string(
            month_unix_sec, TIME_FORMAT_MONTH)
        raise ValueError(
            'Could not download file for station "{0:s}", month {1:s}.'.format(
                station_id, month_string))

    return None


def read_1minute_winds_from_raw_file(text_file_name, utc_offset_hours):
    """Reads 1-minute wind observations from raw file.

    This file should contain 1-minute METARs for one station-month (see
    download_1minute_file).

    :param text_file_name: Path to input file.
    :param utc_offset_hours: Difference between local station time and UTC
        (local minus UTC).
    :return: wind_table: pandas DataFrame with the following columns.
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
    error_checking.assert_is_not_nan(utc_offset_hours)

    unix_times_sec = []
    wind_speeds_m_s01 = []
    wind_directions_deg = []
    wind_gust_speeds_m_s01 = []
    wind_gust_directions_deg = []

    for this_line in open(text_file_name, 'r').readlines():
        this_local_hour_string = (
            this_line[LOCAL_TIME_CHAR_INDICES_1MINUTE_FILE[0]:
                      LOCAL_TIME_CHAR_INDICES_1MINUTE_FILE[1]])
        this_local_date_string = (
            this_line[LOCAL_DATE_CHAR_INDICES_1MINUTE_FILE[0]:
                      LOCAL_DATE_CHAR_INDICES_1MINUTE_FILE[1]])

        this_local_time_string = this_local_date_string + this_local_hour_string
        this_time_unix_sec = _local_time_string_to_unix_sec(
            this_local_time_string, utc_offset_hours)

        (this_wind_speed_m_s01, this_wind_direction_deg,
         this_wind_gust_speed_m_s01,
         this_wind_gust_direction_deg) = _parse_1minute_wind_from_line(
             this_line)

        unix_times_sec.append(this_time_unix_sec)
        wind_speeds_m_s01.append(this_wind_speed_m_s01)
        wind_directions_deg.append(this_wind_direction_deg)
        wind_gust_speeds_m_s01.append(this_wind_gust_speed_m_s01)
        wind_gust_directions_deg.append(this_wind_gust_direction_deg)

    wind_dict = {raw_wind_io.WIND_SPEED_COLUMN: wind_speeds_m_s01,
                 raw_wind_io.WIND_DIR_COLUMN: wind_directions_deg,
                 raw_wind_io.WIND_GUST_SPEED_COLUMN: wind_gust_speeds_m_s01,
                 raw_wind_io.WIND_GUST_DIR_COLUMN: wind_gust_directions_deg,
                 raw_wind_io.TIME_COLUMN: unix_times_sec}

    wind_table = pandas.DataFrame.from_dict(wind_dict)
    wind_table[raw_wind_io.WIND_SPEED_COLUMN] *= KT_TO_METRES_PER_SECOND
    wind_table[raw_wind_io.WIND_GUST_SPEED_COLUMN] *= KT_TO_METRES_PER_SECOND
    return _remove_invalid_wind_rows(wind_table)


def read_5minute_winds_from_raw_file(text_file_name, utc_offset_hours):
    """Reads 5-minute wind observations from raw file.

    This file should contain 5-minute METARs for one station-month (see
    download_5minute_file).

    :param text_file_name: Path to input file.
    :param utc_offset_hours: Difference between local station time and UTC
        (local minus UTC).
    :return: wind_table: pandas DataFrame with the following columns.
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
    error_checking.assert_is_not_nan(utc_offset_hours)

    unix_times_sec = []
    wind_speeds_m_s01 = []
    wind_directions_deg = []
    wind_gust_speeds_m_s01 = []
    wind_gust_directions_deg = []

    for this_line in open(text_file_name, 'r').readlines():
        this_local_time_string = (
            this_line[LOCAL_TIME_CHAR_INDICES_5MINUTE_FILE[0]:
                      LOCAL_TIME_CHAR_INDICES_5MINUTE_FILE[1]])
        this_time_unix_sec = _local_time_string_to_unix_sec(
            this_local_time_string, utc_offset_hours)

        (this_wind_speed_m_s01, this_wind_direction_deg,
         this_wind_gust_speed_m_s01,
         this_wind_gust_direction_deg) = _parse_5minute_wind_from_line(
             this_line)

        unix_times_sec.append(this_time_unix_sec)
        wind_speeds_m_s01.append(this_wind_speed_m_s01)
        wind_directions_deg.append(this_wind_direction_deg)
        wind_gust_speeds_m_s01.append(this_wind_gust_speed_m_s01)
        wind_gust_directions_deg.append(this_wind_gust_direction_deg)

    wind_dict = {raw_wind_io.WIND_SPEED_COLUMN: wind_speeds_m_s01,
                 raw_wind_io.WIND_DIR_COLUMN: wind_directions_deg,
                 raw_wind_io.WIND_GUST_SPEED_COLUMN: wind_gust_speeds_m_s01,
                 raw_wind_io.WIND_GUST_DIR_COLUMN: wind_gust_directions_deg,
                 raw_wind_io.TIME_COLUMN: unix_times_sec}

    wind_table = pandas.DataFrame.from_dict(wind_dict)
    wind_table[raw_wind_io.WIND_SPEED_COLUMN] *= KT_TO_METRES_PER_SECOND
    wind_table[raw_wind_io.WIND_GUST_SPEED_COLUMN] *= KT_TO_METRES_PER_SECOND
    return _remove_invalid_wind_rows(wind_table)


def merge_winds_and_station_metadata(wind_table, station_metadata_table,
                                     station_id):
    """Merges wind data with metadata for observing stations.

    :param wind_table: pandas DataFrame created by
        read_1minute_winds_from_raw_file, read_5minute_winds_from_raw_file, or
        `raw_wind_io.sustained_and_gust_to_uv_max`.
    :param station_metadata_table: pandas DataFrame created by
        read_station_metadata_from_raw_file.
    :param station_id: String ID for station in wind_table.
    :return: wind_table: Same as input, but with the following extra columns.
    wind_table.station_id: String ID for station.
    wind_table.station_name: Verbose name for station.
    wind_table.latitude_deg: Latitude (deg N).
    wind_table.longitude_deg: Longitude (deg E).
    wind_table.elevation_m_asl: Elevation (metres above sea level).
    """

    error_checking.assert_is_string(station_id)

    station_id_list = [station_id] * len(wind_table.index)
    argument_dict = {raw_wind_io.STATION_ID_COLUMN: station_id_list}
    wind_table = wind_table.assign(**argument_dict)

    return wind_table.merge(station_metadata_table[METADATA_COLUMNS_TO_MERGE],
                            on=raw_wind_io.STATION_ID_COLUMN, how='inner')


if __name__ == '__main__':
    # Read metadata from original text file; write to new CSV file.
    STATION_METADATA_TABLE = read_station_metadata_from_raw_file(
        ORIG_METAFILE_NAME)
    raw_wind_io.write_station_metadata_to_processed_file(STATION_METADATA_TABLE,
                                                         NEW_METAFILE_NAME)

    # Download 1-minute METARs and convert file type.
    ORIG_1MINUTE_FILE_NAME = download_1minute_file(
        station_id=ORIG_STATION_ID_1MINUTE,
        month_unix_sec=MONTH_1MINUTE_UNIX_SEC,
        top_local_directory_name=TOP_LOCAL_DIR_NAME_1MINUTE,
        raise_error_if_fails=True)

    THESE_STATION_FLAGS = [
        s == STATION_ID_1MINUTE for s in STATION_METADATA_TABLE[
            raw_wind_io.STATION_ID_COLUMN].values]
    THIS_STATION_INDEX = numpy.where(THESE_STATION_FLAGS)[0][0]
    THIS_UTC_OFFSET_HOURS = STATION_METADATA_TABLE[
        raw_wind_io.UTC_OFFSET_COLUMN].values[THIS_STATION_INDEX]

    WIND_TABLE_1MINUTE = read_1minute_winds_from_raw_file(
        ORIG_1MINUTE_FILE_NAME, THIS_UTC_OFFSET_HOURS)
    WIND_TABLE_1MINUTE = raw_wind_io.sustained_and_gust_to_uv_max(
        WIND_TABLE_1MINUTE)
    WIND_TABLE_1MINUTE = merge_winds_and_station_metadata(
        WIND_TABLE_1MINUTE, STATION_METADATA_TABLE, STATION_ID_1MINUTE)
    print WIND_TABLE_1MINUTE

    raw_wind_io.write_processed_file(WIND_TABLE_1MINUTE, CSV_FILE_NAME_1MINUTE)

    # Download 5-minute METARs and convert file type.
    ORIG_5MINUTE_FILE_NAME = download_5minute_file(
        station_id=ORIG_STATION_ID_5MINUTE,
        month_unix_sec=MONTH_5MINUTE_UNIX_SEC,
        top_local_directory_name=TOP_LOCAL_DIR_NAME_5MINUTE,
        raise_error_if_fails=True)

    THESE_STATION_FLAGS = [
        s == STATION_ID_5MINUTE for s in STATION_METADATA_TABLE[
            raw_wind_io.STATION_ID_COLUMN].values]
    THIS_STATION_INDEX = numpy.where(THESE_STATION_FLAGS)[0][0]
    THIS_UTC_OFFSET_HOURS = STATION_METADATA_TABLE[
        raw_wind_io.UTC_OFFSET_COLUMN].values[THIS_STATION_INDEX]

    WIND_TABLE_5MINUTE = read_5minute_winds_from_raw_file(
        ORIG_5MINUTE_FILE_NAME, THIS_UTC_OFFSET_HOURS)
    WIND_TABLE_5MINUTE = raw_wind_io.sustained_and_gust_to_uv_max(
        WIND_TABLE_5MINUTE)
    WIND_TABLE_5MINUTE = merge_winds_and_station_metadata(
        WIND_TABLE_5MINUTE, STATION_METADATA_TABLE, STATION_ID_5MINUTE)
    print WIND_TABLE_5MINUTE

    raw_wind_io.write_processed_file(WIND_TABLE_5MINUTE, CSV_FILE_NAME_5MINUTE)
