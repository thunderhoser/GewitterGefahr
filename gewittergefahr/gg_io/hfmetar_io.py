"""IO methods for HFMETAR* data.

* HFMETARs = high-frequency (1-minute and 5-minute) meteorological aerodrome
  reports.

HFMETARs are available only for ASOS (Automated Surface Observing System)
stations.
"""

import numpy
import pandas
import time
import calendar
from gewittergefahr.gg_io import myrorss_io
from gewittergefahr.gg_io import raw_wind_io

# TODO(thunderhoser): add error-checking to all methods.
# TODO(thunderhoser): replace main method with named high-level method.

ORIG_METAFILE_NAME = '/localdata/ryan.lagerquist/aasswp/asos-stations.txt'
NEW_METAFILE_NAME = (
    '/localdata/ryan.lagerquist/aasswp/hfmetar_station_metadata.csv')

ORIG_1MINUTE_FILE_NAME = '/localdata/ryan.lagerquist/aasswp/64050KBGD201104.dat'
NEW_1MINUTE_FILE_NAME = (
    '/localdata/ryan.lagerquist/aasswp/hfmetar1_wind_BGD_201104.csv')
STATION_ID_1MINUTE = 'BGD_hfmetar'

ORIG_5MINUTE_FILE_NAME = '/localdata/ryan.lagerquist/aasswp/64010KHKA201103.dat'
NEW_5MINUTE_FILE_NAME = (
    '/localdata/ryan.lagerquist/aasswp/hfmetar5_wind_HKA_201103.csv')
STATION_ID_5MINUTE = 'HKA_hfmetar'

DATA_SOURCE = 'hfmetar'
TIME_FORMAT = '%Y%m%d%H%M'
UTC_OFFSET_COLUMN = 'utc_offset_hours'

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

METADATA_COLUMNS_TO_MERGE = [raw_wind_io.STATION_ID_COLUMN,
                             raw_wind_io.STATION_NAME_COLUMN,
                             raw_wind_io.LATITUDE_COLUMN,
                             raw_wind_io.LONGITUDE_COLUMN,
                             raw_wind_io.ELEVATION_COLUMN]


def _time_string_to_unix_sec(time_string):
    """Converts time from string to Unix format (sec since 0000 UTC 1 Jan 1970).

    :param time_string: Time string (format "yyyymmddHHMM").
    :return: unix_time_sec: Time in Unix format.
    """

    return calendar.timegm(time.strptime(time_string, TIME_FORMAT))


def _local_time_string_to_unix_sec(local_time_string, utc_offset_hours):
    """Converts time from local string to Unix format.

    Unix format = seconds since 0000 UTC 1 Jan 1970.

    :param local_time_string: Local time, formatted as "yyyymmddHHMM".
    :param utc_offset_hours: Difference between UTC and local time zone (local
        minus UTC).
    :return: unix_time_sec: Seconds since 0000 UTC 1 Jan 1970.
    """

    local_time_unix_sec = _time_string_to_unix_sec(local_time_string)
    return local_time_unix_sec - utc_offset_hours * HOURS_TO_SECONDS


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
    except IndexError:
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan
    except ValueError:
        wind_direction_deg = numpy.nan

    try:
        wind_speed_kt = float(words[-3])
    except (ValueError, IndexError):
        wind_speed_kt = numpy.nan

    try:
        wind_gust_direction_deg = float(words[-2])
    except (ValueError, IndexError):
        wind_gust_direction_deg = numpy.nan

    try:
        wind_gust_speed_kt = float(words[-1])
    except (ValueError, IndexError):
        wind_gust_speed_kt = numpy.nan

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


def _remove_invalid_station_metadata(station_metadata_table):
    """Removes invalid metadata for ASOS stations.

    :param station_metadata_table: pandas DataFrame created by
        read_station_metadata_from_text.
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
    """Removes rows with invalid data.

    "Invalid data" means that either sustained wind speed and/or gust speed is
    out of range.  If either wind direction (sustained or gust) is out of range,
    it will simply be replaced with 0 deg (due north), since we don't really
    care about direction.

    :param wind_table: pandas DataFrame created by either
        read_1minute_winds_from_text or read_5minute_winds_from_text.
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


def read_station_metadata_from_text(text_file_name):
    """Reads metadata for ASOS stations from text file.

    This text file is provided by the National Climatic Data Center (NCDC) and
    can be found here: www.ncdc.noaa.gov/homr/file/asos-stations.txt

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

        this_station_id = raw_wind_io.append_source_to_station_id(
            this_line[
            STATION_ID_CHAR_INDICES[0]:STATION_ID_CHAR_INDICES[1]].strip(),
            DATA_SOURCE)

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
                             UTC_OFFSET_COLUMN: utc_offsets_hours}

    station_metadata_table = pandas.DataFrame.from_dict(station_metadata_dict)
    station_metadata_table[raw_wind_io.ELEVATION_COLUMN] *= FEET_TO_METRES
    return _remove_invalid_station_metadata(station_metadata_table)


def read_1minute_winds_from_text(text_file_name, utc_offset_hours):
    """Reads wind data from text file with 1-minute METARs.

    These text files are provided by the National Climatic Data Center (NCDC)
    here: ftp.ncdc.noaa.gov/pub/data/asos-onemin/

    Each file contains data for one station and one month.

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
    return _remove_invalid_wind_data(wind_table)


def read_5minute_winds_from_text(text_file_name, utc_offset_hours):
    """Reads wind data from text file with 5-minute METARs.

    These text files are provided by the National Climatic Data Center (NCDC)
    here: ftp.ncdc.noaa.gov/pub/data/asos-fivemin/

    Each file contains data for one station and one month.

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
    return _remove_invalid_wind_data(wind_table)


def merge_winds_and_metadata(wind_table, station_metadata_table, station_id):
    """Merges wind data with metadata for observing stations.

    :param wind_table: pandas DataFrame created by read_1minute_winds_from_text,
        read_5minute_winds_from_text, or
        `raw_wind_io.sustained_and_gust_to_uv_max`.
    :param station_metadata_table: pandas DataFrame created by
        read_station_metadata_from_text.
    :param station_id: String ID for station in wind_table.
    :return: wind_table: Same as input but with additional columns listed below.
    wind_table.station_id: String ID for station.
    wind_table.station_name: Verbose name for station.
    wind_table.latitude_deg: Latitude (deg N).
    wind_table.longitude_deg: Longitude (deg E).
    wind_table.elevation_m_asl: Elevation (metres above sea level).
    """

    num_observations = len(wind_table.index)
    station_id_list = [station_id] * num_observations

    argument_dict = {raw_wind_io.STATION_ID_COLUMN: station_id_list}
    wind_table = wind_table.assign(**argument_dict)

    return wind_table.merge(station_metadata_table[METADATA_COLUMNS_TO_MERGE],
                            on=raw_wind_io.STATION_ID_COLUMN, how='inner')


if __name__ == '__main__':
    # Read metadata from original text file; write to new CSV file.
    station_metadata_table = read_station_metadata_from_text(ORIG_METAFILE_NAME)
    raw_wind_io.write_station_metadata_to_csv(station_metadata_table,
                                              NEW_METAFILE_NAME)

    # Read 1-minute wind data from original text file.
    these_station_flags = [s == STATION_ID_1MINUTE for s in
                           station_metadata_table[
                               raw_wind_io.STATION_ID_COLUMN].values]
    this_station_index = numpy.where(these_station_flags)[0][0]
    this_utc_offset_hours = station_metadata_table[UTC_OFFSET_COLUMN].values[
        this_station_index]

    wind_table_1minute = read_1minute_winds_from_text(ORIG_1MINUTE_FILE_NAME,
                                                      this_utc_offset_hours)
    wind_table_1minute = raw_wind_io.sustained_and_gust_to_uv_max(
        wind_table_1minute)
    wind_table_1minute = merge_winds_and_metadata(wind_table_1minute,
                                                  station_metadata_table,
                                                  STATION_ID_1MINUTE)
    print wind_table_1minute

    # Write 1-minute wind data to new CSV file.
    raw_wind_io.write_winds_to_csv(wind_table_1minute, NEW_1MINUTE_FILE_NAME)

    # Read 5-minute wind data from original text file.
    these_station_flags = [s == STATION_ID_5MINUTE for s in
                           station_metadata_table[
                               raw_wind_io.STATION_ID_COLUMN].values]
    this_station_index = numpy.where(these_station_flags)[0][0]
    this_utc_offset_hours = station_metadata_table[UTC_OFFSET_COLUMN].values[
        this_station_index]

    wind_table_5minute = read_5minute_winds_from_text(ORIG_5MINUTE_FILE_NAME,
                                                      this_utc_offset_hours)
    wind_table_5minute = raw_wind_io.sustained_and_gust_to_uv_max(
        wind_table_5minute)
    wind_table_5minute = merge_winds_and_metadata(wind_table_5minute,
                                                  station_metadata_table,
                                                  STATION_ID_5MINUTE)
    print wind_table_5minute

    # Write 5-minute wind data to new CSV file.
    raw_wind_io.write_winds_to_csv(wind_table_5minute, NEW_5MINUTE_FILE_NAME)
