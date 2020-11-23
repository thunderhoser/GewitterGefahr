"""Finds gaps (temporal discontinuities) between storm-tracking files."""

import argparse
import os.path
import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
FILE_SIZE_WITHOUT_STORMS_BYTES = 2100

FIRST_SPC_DATE_ARG_NAME = 'first_spc_date_string'
LAST_SPC_DATE_ARG_NAME = 'last_spc_date_string'
TOP_TRACKING_DIR_ARG_NAME = 'top_tracking_dir_name'
TRACKING_SCALE_ARG_NAME = 'tracking_scale_metres2'
DATA_SOURCE_ARG_NAME = 'source_name'
MIN_TIME_DIFF_ARG_NAME = 'min_time_diff_seconds'

SPC_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd". Tracks (and '
    'temporal discontinuities between them) will be sought for all dates from '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_SPC_DATE_ARG_NAME, LAST_SPC_DATE_ARG_NAME)

TOP_TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with tracking files.')

TRACKING_SCALE_HELP_STRING = (
    'Tracking scale (minimum object area).  This is used to find exact files in'
    ' `{0:s}`.'
).format(TOP_TRACKING_DIR_ARG_NAME)

DATA_SOURCE_HELP_STRING = (
    'Data source.  Must be a string in the following list.\n{0:s}'
).format(str(tracking_utils.DATA_SOURCE_NAMES))

MIN_TIME_DIFF_HELP_STRING = (
    'Minimum time difference between successive files.  Any larger difference '
    'is considered a temporal discontinuity.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TOP_TRACKING_DIR_ARG_NAME, type=str, required=True,
    help=TOP_TRACKING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_SCALE_ARG_NAME, type=int, required=False,
    default=echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2,
    help=TRACKING_SCALE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DATA_SOURCE_ARG_NAME, type=str, required=False,
    default=tracking_utils.SEGMOTION_NAME, help=DATA_SOURCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_TIME_DIFF_ARG_NAME, type=int, required=False,
    default=610, help=MIN_TIME_DIFF_HELP_STRING)


def _find_tracking_gaps(
        first_spc_date_string, last_spc_date_string, top_tracking_dir_name,
        tracking_scale_metres2, source_name, min_time_diff_seconds):
    """Finds gaps (temporal discontinuities) between storm-tracking files.

    :param first_spc_date_string: See documentation at top of file.
    :param last_spc_date_string: Same.
    :param top_tracking_dir_name: Same.
    :param tracking_scale_metres2: Same.
    :param source_name: Same.
    :param min_time_diff_seconds: Same.
    """

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    tracking_file_names = []
    unix_times_sec = numpy.array([], dtype=int)
    num_spc_dates = len(spc_date_strings)

    for i in range(num_spc_dates):
        print('Finding tracking files for SPC date "{0:s}"...'.format(
            spc_date_strings[i]
        ))

        these_file_names = tracking_io.find_files_one_spc_date(
            spc_date_string=spc_date_strings[i], source_name=source_name,
            top_tracking_dir_name=top_tracking_dir_name,
            tracking_scale_metres2=tracking_scale_metres2,
            raise_error_if_missing=False
        )[0]

        print(len(these_file_names))

        if not len(these_file_names):
            continue

        these_file_sizes_bytes = numpy.array(
            [os.path.getsize(f) for f in these_file_names], dtype=int
        )
        these_valid_indices = numpy.where(
            these_file_sizes_bytes > FILE_SIZE_WITHOUT_STORMS_BYTES
        )[0]
        these_file_names = [these_file_names[k] for k in these_valid_indices]

        these_unix_times_sec = numpy.array([
            tracking_io.file_name_to_time(f) for f in these_file_names
        ], dtype=int)

        these_sort_indices = numpy.argsort(these_unix_times_sec)
        these_unix_times_sec = these_unix_times_sec[these_sort_indices]
        these_file_names = [these_file_names[k] for k in these_sort_indices]

        tracking_file_names += these_file_names
        unix_times_sec = numpy.concatenate((
            unix_times_sec, these_unix_times_sec
        ))

    time_diffs_seconds = numpy.diff(unix_times_sec)
    time_gap_indices = numpy.where(
        time_diffs_seconds >= min_time_diff_seconds
    )[0]

    num_time_gaps = len(time_gap_indices)

    print((
        '\nThere are {0:d} time gaps (successive files >= {1:d} seconds apart),'
        ' listed below:\n'
    ).format(num_time_gaps, min_time_diff_seconds))

    for i in time_gap_indices:
        this_start_time_string = time_conversion.unix_sec_to_string(
            unix_times_sec[i], TIME_FORMAT)
        this_end_time_string = time_conversion.unix_sec_to_string(
            unix_times_sec[i + 1], TIME_FORMAT)

        print('Gap between {0:s} and {1:s} = {2:d} seconds'.format(
            this_start_time_string, this_end_time_string, time_diffs_seconds[i]
        ))


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _find_tracking_gaps(
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_SPC_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_ARG_NAME),
        top_tracking_dir_name=getattr(
            INPUT_ARG_OBJECT, TOP_TRACKING_DIR_ARG_NAME),
        tracking_scale_metres2=getattr(
            INPUT_ARG_OBJECT, TRACKING_SCALE_ARG_NAME),
        source_name=getattr(INPUT_ARG_OBJECT, DATA_SOURCE_ARG_NAME),
        min_time_diff_seconds=getattr(INPUT_ARG_OBJECT, MIN_TIME_DIFF_ARG_NAME)
    )
