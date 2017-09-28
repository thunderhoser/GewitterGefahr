"""IO methods for reports of damaging SLW* in the Storm Events database.

* SLW = straight-line wind

Raw files are downloaded from here: ftp://ftp.ncdc.noaa.gov/pub/data/swdi/
stormevents/csvfiles/StormEvents_details*.csv.gz

Unfortunately there is no way to script the download, because the file names
contain last-modified dates.  Each gzip file contains only one CSV file, so this
code does not handle the gzip files.  This code assumes that the raw data format
(or "native format") is CSV.
"""

import os.path
import numpy
import pandas
from gewittergefahr.gg_io import raw_wind_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): replace main method with named method.

PATHLESS_RAW_FILE_PREFIX = 'storm_events'
REQUIRED_EVENT_TYPE = 'thunderstorm wind'
HOURS_TO_SECONDS = 3600
KT_TO_METRES_PER_SECOND = 1.852 / 3.6

TIME_COLUMN_ORIG = 'BEGIN_DATE_TIME'
EVENT_TYPE_COLUMN_ORIG = 'EVENT_TYPE'
TIME_ZONE_COLUMN_ORIG = 'CZ_TIMEZONE'
WIND_SPEED_COLUMN_ORIG = 'MAGNITUDE'
LATITUDE_COLUMN_ORIG = 'BEGIN_LAT'
LONGITUDE_COLUMN_ORIG = 'BEGIN_LON'

TIME_ZONE_STRINGS = ['PST', 'PST-8', 'MST', 'MST-7', 'CST', 'CST-6', 'EST',
                     'EST-5', 'AST', 'AST-4']
UTC_OFFSETS_HOURS = numpy.array([-8, -8, -7, -7, -6, -6, -5, -5, -4, -4])

MONTH_NAMES_3LETTERS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                        'Sep', 'Oct', 'Nov', 'Dec']
TIME_FORMAT = '%d-%b-%y %H:%M:%S'

# The following constants are used only in the main method.
ORIG_CSV_FILE_NAME = (
    '/localdata/ryan.lagerquist/software/matlab/madis/raw_files/storm_events/'
    'storm_events2011.csv')
NEW_CSV_FILE_NAME = '/localdata/ryan.lagerquist/aasswp/slw_reports_2011.csv'


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
        "thunderstorm wind," with any combination of capital and lower-case
        letters.
    :return: is_thunderstorm_wind: Boolean flag, either True or False.
    """

    index_of_thunderstorm_wind = event_type_string.lower().find(
        REQUIRED_EVENT_TYPE)
    return index_of_thunderstorm_wind != -1


def _remove_invalid_data(wind_table):
    """Removes any row with invalid data.

    "Invalid data" means that either [a] time is NaN or [b] latitude, longitude,
    or wind speed is out of range.

    :param wind_table: pandas DataFrame created by
        read_slw_reports_from_orig_csv.
    :return: wind_table: Same as input, except that any row with invalid data
        has been removed.
    """

    invalid_flags = numpy.isnan(wind_table[raw_wind_io.TIME_COLUMN].values)
    invalid_indices = numpy.where(invalid_flags)[0]
    wind_table.drop(wind_table.index[invalid_indices], axis=0, inplace=True)

    invalid_indices = raw_wind_io.check_latitudes(
        wind_table[raw_wind_io.LATITUDE_COLUMN].values)
    wind_table.drop(wind_table.index[invalid_indices], axis=0, inplace=True)

    invalid_indices = raw_wind_io.check_longitudes(
        wind_table[raw_wind_io.LONGITUDE_COLUMN].values)
    wind_table.drop(wind_table.index[invalid_indices], axis=0, inplace=True)

    absolute_v_winds_m_s01 = numpy.absolute(
        wind_table[raw_wind_io.V_WIND_COLUMN].values)
    invalid_indices = raw_wind_io.check_wind_speeds(absolute_v_winds_m_s01)
    wind_table.drop(wind_table.index[invalid_indices], axis=0, inplace=True)

    wind_table[raw_wind_io.LONGITUDE_COLUMN] = (
        lng_conversion.convert_lng_positive_in_west(
            wind_table[raw_wind_io.LONGITUDE_COLUMN].values, allow_nan=False))
    return wind_table


def find_local_raw_file(year, directory_name=None, raise_error_if_missing=True):
    """Finds raw file on local machine.

    This file should contain all storm reports for one year.

    :param year: [integer] Will look for file from this year.
    :param directory_name: Name of directory with Storm Events files.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will raise an error.
    :return: raw_file_name: File path.  If raise_error_if_missing = False and
        file is missing, this will be the *expected* path.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    error_checking.assert_is_integer(year)
    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    raw_file_name = '{0:s}/{1:s}{2:s}.csv'.format(
        directory_name, PATHLESS_RAW_FILE_PREFIX, _year_number_to_string(year))

    if raise_error_if_missing and not os.path.isfile(raw_file_name):
        raise ValueError(
            'Cannot find raw file.  Expected at location: ' + raw_file_name)

    return raw_file_name


def read_slw_reports_from_orig_csv(csv_file_name):
    """Reads SLW reports from original CSV file (one in the raw database).

    :param csv_file_name: Path to input file.
    :return: wind_table: pandas DataFrame with the following columns.
    wind_table.latitude_deg: Latitude (deg N).
    wind_table.longitude_deg: Longitude (deg E).
    wind_table.unix_time_sec: Observation time (seconds since 0000 UTC 1 Jan
        1970).
    wind_table.u_wind_m_s01: u-component of wind (m/s), assuming that all
        directions are due north.
    wind_table.v_wind_m_s01: v-component of wind (m/s), assuming that all
        directions are due north.
    """

    error_checking.assert_file_exists(csv_file_name)
    report_table = pandas.read_csv(csv_file_name, header=0, sep=',')

    num_reports = len(report_table.index)
    thunderstorm_wind_flags = numpy.full(num_reports, False, dtype=bool)
    for i in range(num_reports):
        thunderstorm_wind_flags[i] = _is_event_thunderstorm_wind(
            report_table[EVENT_TYPE_COLUMN_ORIG].values[i])

    non_tstorm_wind_indices = (
        numpy.where(numpy.invert(thunderstorm_wind_flags))[0])
    report_table.drop(report_table.index[non_tstorm_wind_indices], axis=0,
                      inplace=True)

    num_reports = len(report_table.index)
    unix_times_sec = numpy.full(num_reports, numpy.nan, dtype=int)
    for i in range(num_reports):
        this_utc_offset_hours = _time_zone_string_to_utc_offset(
            report_table[TIME_ZONE_COLUMN_ORIG].values[i])
        if numpy.isnan(this_utc_offset_hours):
            continue

        unix_times_sec[i] = _local_time_string_to_unix_sec(
            report_table[TIME_COLUMN_ORIG].values[i], this_utc_offset_hours)

    # Wind direction is not included in storm reports, so assume due north.
    u_winds_m_s01 = numpy.full(num_reports, 0.)
    v_winds_m_s01 = -KT_TO_METRES_PER_SECOND * report_table[
        WIND_SPEED_COLUMN_ORIG].values

    wind_dict = {raw_wind_io.TIME_COLUMN: unix_times_sec,
                 raw_wind_io.LATITUDE_COLUMN: report_table[
                     LATITUDE_COLUMN_ORIG].values,
                 raw_wind_io.LONGITUDE_COLUMN: report_table[
                     LONGITUDE_COLUMN_ORIG].values,
                 raw_wind_io.U_WIND_COLUMN: u_winds_m_s01,
                 raw_wind_io.V_WIND_COLUMN: v_winds_m_s01}

    wind_table = pandas.DataFrame.from_dict(wind_dict)
    return _remove_invalid_data(wind_table)


if __name__ == '__main__':
    WIND_TABLE = read_slw_reports_from_orig_csv(ORIG_CSV_FILE_NAME)
    print WIND_TABLE

    raw_wind_io.write_winds_to_csv(WIND_TABLE, NEW_CSV_FILE_NAME)
