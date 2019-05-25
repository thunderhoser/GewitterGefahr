"""Methods for handling time periods."""

import numpy
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import error_checking


def range_and_interval_to_list(start_time_unix_sec=None, end_time_unix_sec=None,
                               time_interval_sec=None, include_endpoint=True):
    """Converts time period from range and interval to list of exact times.

    N = number of exact times

    :param start_time_unix_sec: Start time (Unix format).
    :param end_time_unix_sec: End time (Unix format).
    :param time_interval_sec: Interval (seconds) between successive exact times.
    :param include_endpoint: Boolean flag.  If True, endpoint will be included
        in list of time steps.  If False, endpoint will be excluded.
    :return: unix_times_sec: length-N numpy array of exact times (Unix format).
    """

    error_checking.assert_is_integer(start_time_unix_sec)
    error_checking.assert_is_not_nan(start_time_unix_sec)
    error_checking.assert_is_integer(end_time_unix_sec)
    error_checking.assert_is_not_nan(end_time_unix_sec)
    error_checking.assert_is_integer(time_interval_sec)
    error_checking.assert_is_boolean(include_endpoint)

    if include_endpoint:
        error_checking.assert_is_geq(end_time_unix_sec, start_time_unix_sec)
    else:
        error_checking.assert_is_greater(end_time_unix_sec, start_time_unix_sec)

    start_time_unix_sec = int(rounder.floor_to_nearest(
        float(start_time_unix_sec), time_interval_sec
    ))
    end_time_unix_sec = int(rounder.ceiling_to_nearest(
        float(end_time_unix_sec), time_interval_sec
    ))

    if not include_endpoint:
        end_time_unix_sec -= time_interval_sec

    num_time_steps = 1 + int(numpy.round(
        (end_time_unix_sec - start_time_unix_sec) / time_interval_sec
    ))

    return numpy.linspace(start_time_unix_sec, end_time_unix_sec,
                          num=num_time_steps, dtype=int)


def time_and_period_length_to_range(unix_time_sec, period_length_sec):
    """Converts single time and period length to range (start/end of period).

    :param unix_time_sec: Single time (Unix format).
    :param period_length_sec: Length of time period (seconds).
    :return: start_time_unix_sec: Beginning of time period (Unix format).
    :return: end_time_unix_sec: End of time period (Unix format).
    """

    error_checking.assert_is_integer(unix_time_sec)
    error_checking.assert_is_not_nan(unix_time_sec)
    error_checking.assert_is_integer(period_length_sec)

    start_time_unix_sec = int(rounder.floor_to_nearest(
        float(unix_time_sec), period_length_sec
    ))

    return start_time_unix_sec, start_time_unix_sec + period_length_sec


def time_and_period_length_and_interval_to_list(unix_time_sec=None,
                                                period_length_sec=None,
                                                time_interval_sec=None,
                                                include_endpoint=True):
    """Converts single time, period length, and interval to list of exact times.

    :param unix_time_sec: Single time (Unix format).
    :param period_length_sec: Length of time period (seconds).
    :param time_interval_sec: Interval (seconds) between successive exact times.
    :param include_endpoint: Boolean flag.  If True, endpoint will be included
        in list of time steps.  If False, endpoint will be excluded.
    :return: unix_times_sec: length-N numpy array of exact times (Unix format).
    """

    start_time_unix_sec, end_time_unix_sec = time_and_period_length_to_range(
        unix_time_sec, period_length_sec)

    return range_and_interval_to_list(
        start_time_unix_sec=start_time_unix_sec,
        end_time_unix_sec=end_time_unix_sec,
        time_interval_sec=time_interval_sec, include_endpoint=include_endpoint)
