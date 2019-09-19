"""Subsets data by time."""

import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

NUM_MONTHS_IN_YEAR = 12
NUM_HOURS_IN_DAY = 24

VALID_MONTH_COUNTS = numpy.array([1, 3], dtype=int)
VALID_HOUR_COUNTS = numpy.array([1, 3, 6], dtype=int)


def get_monthly_chunks(num_months_per_chunk, verbose):
    """Returns list of months in each chunk.

    :param num_months_per_chunk: Number of months per chunk.
    :param verbose: Boolean flag.  If True, will print messages to command
        window.
    :return: chunk_to_months_dict: Dictionary, where each key is an index and
        the corresponding value is a 1-D numpy array of months (from 1...12).
    :raises: ValueError: if `num_months_per_chunk not in VALID_MONTH_COUNTS`.
    """

    error_checking.assert_is_integer(num_months_per_chunk)
    error_checking.assert_is_boolean(verbose)

    if num_months_per_chunk not in VALID_MONTH_COUNTS:
        error_string = (
            '\n{0:s}\nValid numbers of months per chunk (listed above) do not '
            'include {1:d}.'
        ).format(str(VALID_MONTH_COUNTS), num_months_per_chunk)

        raise ValueError(error_string)

    chunk_to_months_dict = dict()

    if num_months_per_chunk == 1:
        for i in range(NUM_MONTHS_IN_YEAR):
            chunk_to_months_dict[i] = numpy.array([i + 1], dtype=int)
    else:
        chunk_to_months_dict[0] = numpy.array([12, 1, 2], dtype=int)
        chunk_to_months_dict[1] = numpy.array([3, 4, 5], dtype=int)
        chunk_to_months_dict[2] = numpy.array([6, 7, 8], dtype=int)
        chunk_to_months_dict[3] = numpy.array([9, 10, 11], dtype=int)

    if not verbose:
        return chunk_to_months_dict

    num_chunks = len(chunk_to_months_dict.keys())
    for i in range(num_chunks):
        print('Months in {0:d}th chunk = {1:s}'.format(
            i + 1, str(chunk_to_months_dict[i])
        ))

    return chunk_to_months_dict


def get_hourly_chunks(num_hours_per_chunk, verbose):
    """Returns list of hours in each chunk.

    :param num_hours_per_chunk: Number of hours per chunk.
    :param verbose: Boolean flag.  If True, will print messages to command
        window.
    :return: chunk_to_hours_dict: Dictionary, where each key is an index and
        the corresponding value is a 1-D numpy array of hours (from 0...23).
    :raises: ValueError: if `num_hours_per_chunk not in VALID_HOUR_COUNTS`.
    """

    error_checking.assert_is_integer(num_hours_per_chunk)
    error_checking.assert_is_boolean(verbose)

    if num_hours_per_chunk not in VALID_HOUR_COUNTS:
        error_string = (
            '\n{0:s}\nValid numbers of hours per chunk (listed above) do not '
            'include {1:d}.'
        ).format(str(VALID_HOUR_COUNTS), num_hours_per_chunk)

        raise ValueError(error_string)

    chunk_to_hours_dict = dict()
    num_hourly_chunks = int(numpy.round(
        NUM_HOURS_IN_DAY / num_hours_per_chunk
    ))

    for i in range(num_hourly_chunks):
        chunk_to_hours_dict[i] = numpy.linspace(
            i * num_hours_per_chunk, (i + 1) * num_hours_per_chunk - 1,
            num=num_hours_per_chunk, dtype=int
        )

        if not verbose:
            continue

        print('Hours in {0:d}th chunk = {1:s}'.format(
            i + 1, str(chunk_to_hours_dict[i])
        ))

    return chunk_to_hours_dict


def get_events_in_months(desired_months, verbose, event_months=None,
                         event_times_unix_sec=None):
    """Finds events in desired months.

    If `event_months is None`, `event_times_unix_sec` will be used.

    :param desired_months: 1-D numpy array of desired months (range 1...12).
    :param verbose: Boolean flag.  If True, will print messages to command
        window.
    :param event_months: 1-D numpy array of event months (range 1...12).
    :param event_times_unix_sec: 1-D numpy array of event times.
    :return: desired_event_indices: 1-D numpy array with indices of events in
        desired months.
    :return: event_months: See input doc.
    """

    if event_months is None:
        error_checking.assert_is_numpy_array(
            event_times_unix_sec, num_dimensions=1)

        event_months = numpy.array([
            int(time_conversion.unix_sec_to_string(t, '%m'))
            for t in event_times_unix_sec
        ], dtype=int)

    error_checking.assert_is_integer_numpy_array(event_months)
    error_checking.assert_is_numpy_array(event_months, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(event_months, 1)
    error_checking.assert_is_leq_numpy_array(event_months, NUM_MONTHS_IN_YEAR)

    error_checking.assert_is_integer_numpy_array(desired_months)
    error_checking.assert_is_numpy_array(desired_months, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(desired_months, 1)
    error_checking.assert_is_leq_numpy_array(desired_months, NUM_MONTHS_IN_YEAR)

    error_checking.assert_is_boolean(verbose)

    desired_event_flags = numpy.array(
        [m in desired_months for m in event_months], dtype=bool
    )
    desired_event_indices = numpy.where(desired_event_flags)[0]

    if not verbose:
        return desired_event_indices, event_months

    print('{0:d} of {1:d} events are in months {2:s}!'.format(
        len(desired_event_indices), len(event_months), str(desired_months)
    ))

    return desired_event_indices, event_months


def get_events_in_hours(desired_hours, verbose, event_hours=None,
                        event_times_unix_sec=None):
    """Finds events in desired hours.

    If `event_hours is None`, `event_times_unix_sec` will be used.

    :param desired_hours: 1-D numpy array of desired hours (range 0...23).
    :param verbose: Boolean flag.  If True, will print messages to command
        window.
    :param event_hours: 1-D numpy array of event hours (range 0...23).
    :param event_times_unix_sec: 1-D numpy array of event times.
    :return: desired_event_indices: 1-D numpy array with indices of events in
        desired hours.
    """

    if event_hours is None:
        error_checking.assert_is_numpy_array(
            event_times_unix_sec, num_dimensions=1)

        event_hours = numpy.array([
            int(time_conversion.unix_sec_to_string(t, '%H'))
            for t in event_times_unix_sec
        ], dtype=int)

    error_checking.assert_is_integer_numpy_array(event_hours)
    error_checking.assert_is_numpy_array(event_hours, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(event_hours, 0)
    error_checking.assert_is_less_than_numpy_array(
        event_hours, NUM_HOURS_IN_DAY)

    error_checking.assert_is_integer_numpy_array(desired_hours)
    error_checking.assert_is_numpy_array(desired_hours, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(desired_hours, 0)
    error_checking.assert_is_less_than_numpy_array(
        desired_hours, NUM_HOURS_IN_DAY)

    error_checking.assert_is_boolean(verbose)

    desired_event_flags = numpy.array(
        [m in desired_hours for m in event_hours], dtype=bool
    )
    desired_event_indices = numpy.where(desired_event_flags)[0]

    if not verbose:
        return desired_event_indices, event_hours

    print('{0:d} of {1:d} events are in hours {2:s}!'.format(
        len(desired_event_indices), len(event_hours), str(desired_hours)
    ))

    return desired_event_indices, event_hours
