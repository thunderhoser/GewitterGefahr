"""IO methods for tornado and SLTW* reports from the Storm Events database.

* SLTW = straight-line thunderstorm wind

Raw files are downloaded from: ftp://ftp.ncdc.noaa.gov/pub/data/swdi/
stormevents/csvfiles/StormEvents_details*.csv.gz
"""

import os.path
import numpy
import pandas
from gewittergefahr.gg_io import raw_wind_io
from gewittergefahr.gg_io import tornado_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): replace main method with named method.

PATHLESS_FILE_PREFIX = 'storm_events'
FILE_EXTENSION = '.csv'

HOURS_TO_SECONDS = 3600
KT_TO_METRES_PER_SECOND = 1.852 / 3.6
FEET_TO_METRES = 1. / 3.2808

TORNADO_EVENT_TYPE = 'tornado'
DAMAGING_WIND_EVENT_TYPE = 'thunderstorm wind'

START_TIME_COLUMN_ORIG = 'BEGIN_DATE_TIME'
END_TIME_COLUMN_ORIG = 'END_DATE_TIME'
EVENT_TYPE_COLUMN_ORIG = 'EVENT_TYPE'
TIME_ZONE_COLUMN_ORIG = 'CZ_TIMEZONE'
START_LATITUDE_COLUMN_ORIG = 'BEGIN_LAT'
START_LONGITUDE_COLUMN_ORIG = 'BEGIN_LON'
END_LATITUDE_COLUMN_ORIG = 'END_LAT'
END_LONGITUDE_COLUMN_ORIG = 'END_LON'

WIND_SPEED_COLUMN_ORIG = 'MAGNITUDE'
TORNADO_RATING_COLUMN_ORIG = 'TOR_F_SCALE'
TORNADO_WIDTH_COLUMN_ORIG = 'TOR_WIDTH'

TIME_ZONE_STRINGS = ['PST', 'PST-8', 'MST', 'MST-7', 'CST', 'CST-6', 'EST',
                     'EST-5', 'AST', 'AST-4']
UTC_OFFSETS_HOURS = numpy.array([-8, -8, -7, -7, -6, -6, -5, -5, -4, -4])

MONTH_NAMES_3LETTERS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                        'Sep', 'Oct', 'Nov', 'Dec']
TIME_FORMAT = '%d-%b-%y %H:%M:%S'

# The following constants are used only in the main method.
STORM_EVENT_FILE_NAME = (
    '/localdata/ryan.lagerquist/gewittergefahr_data/wind_observations/'
    'storm_events/storm_events2011.csv')
PROCESSED_WIND_FILE_NAME = (
    '/localdata/ryan.lagerquist/gewittergefahr_data/wind_observations/'
    'storm_events/sltw_reports_2011.csv')
PROCESSED_TORNADO_FILE_NAME = (
    '/localdata/ryan.lagerquist/gewittergefahr_data/wind_observations/'
    'storm_events/tornado_reports_2011.csv')


def _year_number_to_string(year):
    """Converts year from number to string.

    :param year: Integer year.
    :return: year_string: String in format "yyyy" (with leading zeros if
        necessary).
    """

    return '{0:04d}'.format(int(year))


def _time_zone_string_to_utc_offset(time_zone_string):
    """Converts time zone from string to UTC offset.

    :param time_zone_string: String describing time zone (examples: "PST",
        "MST", etc.).
    :return: utc_offset_hours: Local time minus UTC.
    """

    time_zone_string = time_zone_string.strip().upper()
    tz_string_flags = [s == time_zone_string for s in TIME_ZONE_STRINGS]
    if not any(tz_string_flags):
        return numpy.nan

    tz_string_index = numpy.where(tz_string_flags)[0][0]
    return UTC_OFFSETS_HOURS[tz_string_index]


def _capitalize_months(orig_string):
    """Converts string to all lower-case, except first letter of each month.

    :param orig_string: Original string.
    :return: new_string: New string.
    """

    new_string = orig_string.lower()
    for i in range(len(MONTH_NAMES_3LETTERS)):
        new_string = new_string.replace(MONTH_NAMES_3LETTERS[i].lower(),
                                        MONTH_NAMES_3LETTERS[i])

    return new_string


def _time_string_to_unix_sec(time_string):
    """Converts time from string to Unix format.

    :param time_string: Time string (format "dd-mmm-yy HH:MM:SS").
    :return: unix_time_sec: Time in Unix format.
    """

    time_string = _capitalize_months(time_string)
    return time_conversion.string_to_unix_sec(time_string, TIME_FORMAT)


def _local_time_string_to_unix_sec(local_time_string, utc_offset_hours):
    """Converts time from local string to Unix format.

    :param local_time_string: Local time (format "dd-mmm-yy HH:MM:SS").
    :param utc_offset_hours: Local time minus UTC.
    :return: unix_time_sec: UTC time in Unix format.
    """

    return _time_string_to_unix_sec(local_time_string) - (
        utc_offset_hours * HOURS_TO_SECONDS)


def _is_event_thunderstorm_wind(event_type_string):
    """Determines whether or not event type is thunderstorm wind.

    :param event_type_string: String description of event.  Must contain
        "thunderstorm wind", with any combination of capital and lower-case
        letters.
    :return: is_thunderstorm_wind: Boolean flag.
    """

    index_of_thunderstorm_wind = event_type_string.lower().find(
        DAMAGING_WIND_EVENT_TYPE)
    return index_of_thunderstorm_wind != -1


def _is_event_tornado(event_type_string):
    """Determines whether or not event type is tornado.

    :param event_type_string: String description of event.  Must contain
        "tornado", with any combination of capital and lower-case letters.
    :return: is_tornado: Boolean flag.
    """

    index_of_tornado = event_type_string.lower().find(TORNADO_EVENT_TYPE)
    return index_of_tornado != -1


def _create_fake_station_ids_for_wind(num_reports):
    """Creates a fake station ID for each wind report.

    All other wind datasets (high-frequency METARs, MADIS, and the Oklahoma
    Mesonet) have station IDs.  This makes Storm Events look like the other
    datasets (i.e., ensures that all datasets have the same columns), which
    allows them to be easily merged.

    N = number of wind reports

    :param num_reports: Number of wind reports.
    :return: station_ids: length-N list of string IDs.
    """

    numeric_station_ids = numpy.linspace(
        0, num_reports - 1, num=num_reports, dtype=int)

    return [raw_wind_io.append_source_to_station_id(
        '{0:06d}'.format(i),
        primary_source=raw_wind_io.STORM_EVENTS_DATA_SOURCE)
            for i in numeric_station_ids]


def _remove_invalid_wind_reports(wind_table):
    """Removes wind reports with invalid v-wind, latitude, longitude, or time.

    :param wind_table: pandas DataFrame created by
        `read_thunderstorm_wind_reports`.
    :return: wind_table: Same as input, except that some rows may be gone.
    """

    return raw_wind_io.remove_invalid_rows(
        wind_table, check_v_wind_flag=True, check_lat_flag=True,
        check_lng_flag=True, check_time_flag=True)


def find_file(year, directory_name, raise_error_if_missing=True):
    """Finds Storm Events file.

    This file should contain all storm reports for one year.

    :param year: Year (integer).
    :param directory_name: Name of directory with Storm Events files.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        raise_error_if_missing = True, this method will error out.
    :return: storm_event_file_name: Path to Storm Events file.  If file is
        missing and raise_error_if_missing = False, this will be the *expected*
        path.
    :raises: ValueError: if file is missing and raise_error_if_missing = True.
    """

    error_checking.assert_is_integer(year)
    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    storm_event_file_name = '{0:s}/{1:s}{2:s}{3:s}'.format(
        directory_name, PATHLESS_FILE_PREFIX, _year_number_to_string(year),
        FILE_EXTENSION)

    if raise_error_if_missing and not os.path.isfile(storm_event_file_name):
        error_string = (
            'Cannot find Storm Events file.  Expected at: {0:s}'.format(
                storm_event_file_name))
        raise ValueError(error_string)

    return storm_event_file_name


def read_thunderstorm_wind_reports(csv_file_name):
    """Reads thunderstorm-wind reports from file.

    This file should contain all storm reports for one year.

    :param csv_file_name: Path to input file.
    :return: wind_table: pandas DataFrame with the following columns.
    wind_table.latitude_deg: Latitude (deg N).
    wind_table.longitude_deg: Longitude (deg E).
    wind_table.unix_time_sec: Valid time.
    wind_table.u_wind_m_s01: u-component of wind (metres per second).
    wind_table.v_wind_m_s01: v-component of wind (metres per second).
    """

    error_checking.assert_file_exists(csv_file_name)
    storm_event_table = pandas.read_csv(csv_file_name, header=0, sep=',')

    thunderstorm_wind_flags = numpy.array(
        [_is_event_thunderstorm_wind(s)
         for s in storm_event_table[EVENT_TYPE_COLUMN_ORIG].values])

    bad_rows = numpy.where(numpy.invert(thunderstorm_wind_flags))[0]
    storm_event_table.drop(
        storm_event_table.index[bad_rows], axis=0, inplace=True)

    num_reports = len(storm_event_table.index)
    unix_times_sec = numpy.full(num_reports, -1, dtype=int)
    for i in range(num_reports):
        this_utc_offset_hours = _time_zone_string_to_utc_offset(
            storm_event_table[TIME_ZONE_COLUMN_ORIG].values[i])
        if numpy.isnan(this_utc_offset_hours):
            continue

        unix_times_sec[i] = _local_time_string_to_unix_sec(
            storm_event_table[START_TIME_COLUMN_ORIG].values[i],
            this_utc_offset_hours)

    wind_speeds_m_s01 = KT_TO_METRES_PER_SECOND * storm_event_table[
        WIND_SPEED_COLUMN_ORIG].values
    wind_directions_deg = numpy.full(
        num_reports, raw_wind_io.WIND_DIR_DEFAULT_DEG)
    u_winds_m_s01, v_winds_m_s01 = raw_wind_io.speed_and_direction_to_uv(
        wind_speeds_m_s01, wind_directions_deg)

    station_ids = _create_fake_station_ids_for_wind(num_reports)

    wind_dict = {
        raw_wind_io.TIME_COLUMN: unix_times_sec,
        raw_wind_io.LATITUDE_COLUMN:
            storm_event_table[START_LATITUDE_COLUMN_ORIG].values,
        raw_wind_io.LONGITUDE_COLUMN:
            storm_event_table[START_LONGITUDE_COLUMN_ORIG].values,
        raw_wind_io.STATION_ID_COLUMN: station_ids,
        raw_wind_io.STATION_NAME_COLUMN: station_ids,
        raw_wind_io.ELEVATION_COLUMN: numpy.full(num_reports, numpy.nan),
        raw_wind_io.U_WIND_COLUMN: u_winds_m_s01,
        raw_wind_io.V_WIND_COLUMN: v_winds_m_s01}

    wind_table = pandas.DataFrame.from_dict(wind_dict)
    return _remove_invalid_wind_reports(wind_table)


def read_tornado_reports(csv_file_name):
    """Reads tornado reports from file.

    This file should contain all storm reports for one year.

    :param csv_file_name: Path to input file.
    :return: tornado_table: pandas DataFrame with the following columns.
    tornado_table.start_time_unix_sec: Start time.
    tornado_table.end_time_unix_sec: End time.
    tornado_table.start_latitude_deg: Latitude (deg N) of start point.
    tornado_table.start_longitude_deg: Longitude (deg E) of start point.
    tornado_table.end_latitude_deg: Latitude (deg N) of end point.
    tornado_table.end_longitude_deg: Longitude (deg E) of end point.
    tornado_table.fujita_rating: F-scale or EF-scale rating (integer from
        0...5).
    tornado_table.width_metres: Tornado width (metres).
    """

    error_checking.assert_file_exists(csv_file_name)
    storm_event_table = pandas.read_csv(csv_file_name, header=0, sep=',')

    tornado_flags = numpy.array(
        [_is_event_tornado(s)
         for s in storm_event_table[EVENT_TYPE_COLUMN_ORIG].values])

    bad_rows = numpy.where(numpy.invert(tornado_flags))[0]
    storm_event_table.drop(
        storm_event_table.index[bad_rows], axis=0, inplace=True)

    num_reports = len(storm_event_table.index)
    start_times_unix_sec = numpy.full(num_reports, -1, dtype=int)
    end_times_unix_sec = numpy.full(num_reports, -1, dtype=int)

    for i in range(num_reports):
        this_utc_offset_hours = _time_zone_string_to_utc_offset(
            storm_event_table[TIME_ZONE_COLUMN_ORIG].values[i])
        if numpy.isnan(this_utc_offset_hours):
            continue

        start_times_unix_sec[i] = _local_time_string_to_unix_sec(
            storm_event_table[START_TIME_COLUMN_ORIG].values[i],
            this_utc_offset_hours)
        end_times_unix_sec[i] = _local_time_string_to_unix_sec(
            storm_event_table[END_TIME_COLUMN_ORIG].values[i],
            this_utc_offset_hours)

    tornado_dict = {
        tornado_io.START_TIME_COLUMN: start_times_unix_sec,
        tornado_io.END_TIME_COLUMN: end_times_unix_sec,
        tornado_io.START_LAT_COLUMN:
            storm_event_table[START_LATITUDE_COLUMN_ORIG].values,
        tornado_io.START_LNG_COLUMN:
            storm_event_table[START_LONGITUDE_COLUMN_ORIG].values,
        tornado_io.END_LAT_COLUMN:
            storm_event_table[END_LATITUDE_COLUMN_ORIG].values,
        tornado_io.END_LNG_COLUMN:
            storm_event_table[END_LONGITUDE_COLUMN_ORIG].values,
        tornado_io.FUJITA_RATING_COLUMN:
            storm_event_table[TORNADO_RATING_COLUMN_ORIG].values,
        tornado_io.WIDTH_COLUMN: FEET_TO_METRES * storm_event_table[
            TORNADO_WIDTH_COLUMN_ORIG].values
    }

    tornado_table = pandas.DataFrame.from_dict(tornado_dict)
    return tornado_io.remove_invalid_reports(tornado_table)


if __name__ == '__main__':
    WIND_TABLE = read_thunderstorm_wind_reports(STORM_EVENT_FILE_NAME)
    print(WIND_TABLE)

    raw_wind_io.write_processed_file(WIND_TABLE, PROCESSED_WIND_FILE_NAME)
    print('\n')

    TORNADO_TABLE = read_tornado_reports(STORM_EVENT_FILE_NAME)
    print(TORNADO_TABLE)

    tornado_io.write_processed_file(TORNADO_TABLE, PROCESSED_TORNADO_FILE_NAME)
