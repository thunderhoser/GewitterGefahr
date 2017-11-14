"""Methods for time conversion.

--- DEFINITIONS ---

SPC = Storm Prediction Center

SPC date = a 24-hour period running from 1200-1200 UTC.  If time is discretized
in seconds, the period runs from 120000-115959 UTC.  This is unlike a human
date, which runs from 0000-0000 UTC (or 000000-235959 UTC).
"""

import time
import calendar
import numpy
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import error_checking

SPC_DATE_FORMAT = '%Y%m%d'
HOURS_TO_SECONDS = 3600
DAYS_TO_SECONDS = 86400
SECONDS_INTO_SPC_DATE_DEFAULT = 18 * HOURS_TO_SECONDS

MIN_SECONDS_INTO_SPC_DATE = 12 * HOURS_TO_SECONDS
MAX_SECONDS_INTO_SPC_DATE = (36 * HOURS_TO_SECONDS) - 1


def string_to_unix_sec(time_string, time_directive):
    """Converts time from string to Unix format.

    Unix format = seconds since 0000 UTC 1 Jan 1970.

    :param time_string: Time string.
    :param time_directive: Format of time string (examples: "%Y%m%d" if string
        is "yyyymmdd", "%Y-%m-%d-%H%M%S" if string is "yyyy-mm-dd-HHMMSS",
        etc.).
    :return: unix_time_sec: Time in Unix format.
    """

    error_checking.assert_is_string(time_string)
    error_checking.assert_is_string(time_directive)
    return calendar.timegm(time.strptime(time_string, time_directive))


def unix_sec_to_string(unix_time_sec, time_directive):
    """Converts time from Unix format to string.

    Unix format = seconds since 0000 UTC 1 Jan 1970.

    :param unix_time_sec: Time in Unix format.
    :param time_directive: Format of time string (examples: "%Y%m%d" if string
        is "yyyymmdd", "%Y-%m-%d-%H%M%S" if string is "yyyy-mm-dd-HHMMSS",
        etc.).
    :return: time_string: Time string.
    """

    error_checking.assert_is_integer(unix_time_sec)
    error_checking.assert_is_string(time_directive)
    return time.strftime(time_directive, time.gmtime(unix_time_sec))


def time_to_spc_date_unix_sec(unix_time_sec):
    """Converts time to SPC date (both in Unix format).

    :param unix_time_sec: Time in Unix format.
    :return: spc_date_unix_sec: SPC date in Unix format.  If the SPC date is
        "Oct 28 2017" (120000 UTC 28 Oct - 115959 UTC 29 Oct 2017),
        spc_date_unix_sec will be 180000 UTC 28 Oct 2017.  In general,
        spc_date_unix_sec will be 6 hours into the SPC date.
    """

    error_checking.assert_is_integer(unix_time_sec)
    return int(SECONDS_INTO_SPC_DATE_DEFAULT + rounder.floor_to_nearest(
        unix_time_sec - DAYS_TO_SECONDS / 2, DAYS_TO_SECONDS))


def time_to_spc_date_string(unix_time_sec):
    """Converts time in Unix format to SPC date in string format.

    :param unix_time_sec: Time in Unix format.
    :return: spc_date_string: SPC date in format "yyyymmdd".
    """

    error_checking.assert_is_integer(unix_time_sec)
    return unix_sec_to_string(
        unix_time_sec - DAYS_TO_SECONDS / 2, SPC_DATE_FORMAT)


def spc_date_string_to_unix_sec(spc_date_string):
    """Converts SPC date from string to Unix format.

    :param spc_date_string: SPC date in format "yyyymmdd".
    :return: spc_date_unix_sec: SPC date in Unix format.  If the SPC date is
        "Oct 28 2017" (120000 UTC 28 Oct - 115959 UTC 29 Oct 2017),
        spc_date_unix_sec will be 180000 UTC 28 Oct 2017.  In general,
        spc_date_unix_sec will be 6 hours into the SPC date.
    """

    return SECONDS_INTO_SPC_DATE_DEFAULT + string_to_unix_sec(
        spc_date_string, SPC_DATE_FORMAT)


def is_time_in_spc_date(unix_time_sec, spc_date_string):
    """Determines whether or not time is in SPC date.

    :param unix_time_sec: Time in Unix format.
    :param spc_date_string: SPC date in format "yyyymmdd".
    :return: time_in_spc_date_flag: Boolean flag.
    """

    min_time_unix_sec = MIN_SECONDS_INTO_SPC_DATE + string_to_unix_sec(
        spc_date_string, SPC_DATE_FORMAT)
    max_time_unix_sec = MAX_SECONDS_INTO_SPC_DATE + string_to_unix_sec(
        spc_date_string, SPC_DATE_FORMAT)

    error_checking.assert_is_integer(unix_time_sec)
    error_checking.assert_is_not_nan(unix_time_sec)

    return numpy.logical_and(unix_time_sec >= min_time_unix_sec,
                             unix_time_sec <= max_time_unix_sec)
