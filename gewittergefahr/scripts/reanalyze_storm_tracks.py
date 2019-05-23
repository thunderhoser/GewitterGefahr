"""Reanalyzes storm tracks (preferably over many SPC dates)."""

import argparse
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import time_conversion

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
MYRORSS_START_TIME_STRING = '1990-01-01-000000'
MYRORSS_END_TIME_STRING = '2012-01-01-000813'

INPUT_DIR_ARG_NAME = 'input_tracking_dir_name'
FIRST_SPC_DATE_ARG_NAME = 'first_spc_date_string'
LAST_SPC_DATE_ARG_NAME = 'last_spc_date_string'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
MAX_VELOCITY_DIFF_ARG_NAME = 'max_velocity_diff_m_s01'
MAX_LINK_DISTANCE_ARG_NAME = 'max_link_distance_m_s01'
MAX_JOIN_TIME_ARG_NAME = 'max_join_time_seconds'
MAX_JOIN_ERROR_ARG_NAME = 'max_join_error_m_s01'
MIN_DURATION_ARG_NAME = 'min_duration_seconds'
OUTPUT_DIR_ARG_NAME = 'output_tracking_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory with original tracks (before reanalysis).  '
    'Files therein will be found by `storm_tracking_io.find_processed_file` and'
    ' read by `storm_tracking_io.read_processed_file`.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Tracks will be reanalyzed for the period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_SPC_DATE_ARG_NAME, LAST_SPC_DATE_ARG_NAME)

FIRST_TIME_HELP_STRING = (
    'First time in period (format "yyyy-mm-dd-HHMMSS").  If empty, defaults to '
    'first time on first SPC date.'
)

LAST_TIME_HELP_STRING = (
    'Last time in period (format "yyyy-mm-dd-HHMMSS").  If empty, defaults to '
    'last time on last SPC date.')

MAX_VELOCITY_DIFF_HELP_STRING = (
    'Used only to bridge the gap between two SPC dates.  See '
    '`echo_top_tracking._link_local_maxima_in_time` for details.')

MAX_LINK_DISTANCE_HELP_STRING = (
    'Used only to bridge the gap between two SPC dates.  See '
    '`echo_top_tracking._link_local_maxima_in_time` for details.')

MAX_JOIN_TIME_HELP_STRING = (
    'Max time difference (between end of early track and beginning of late '
    'track) for joining collinear tracks.')

MAX_JOIN_ERROR_HELP_STRING = (
    'Max extrapolation error (between end of early track and beginning of late '
    'track) for joining collinear tracks.')

MIN_DURATION_HELP_STRING = (
    'Minimum storm duration.  Shorter-lived storms will be thrown out.')

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level directory for new tracks (after reanalysis).  Files will'
    ' be written by `storm_tracking_io.write_processed_file`, to locations '
    'therein determined by `storm_tracking_io.find_processed_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=False, default='',
    help=FIRST_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=False, default='',
    help=LAST_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_VELOCITY_DIFF_ARG_NAME, type=float, required=False,
    default=echo_top_tracking.DEFAULT_MAX_VELOCITY_DIFF_M_S01,
    help=MAX_VELOCITY_DIFF_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LINK_DISTANCE_ARG_NAME, type=float, required=False,
    default=echo_top_tracking.DEFAULT_MAX_LINK_DISTANCE_M_S01,
    help=MAX_LINK_DISTANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_JOIN_TIME_ARG_NAME, type=int, required=False,
    default=echo_top_tracking.DEFAULT_MAX_JOIN_TIME_SEC,
    help=MAX_JOIN_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_JOIN_ERROR_ARG_NAME, type=float, required=False,
    default=echo_top_tracking.DEFAULT_MAX_JOIN_ERROR_M_S01,
    help=MAX_JOIN_ERROR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_DURATION_ARG_NAME, type=int, required=False,
    default=echo_top_tracking.DEFAULT_MIN_REANALYZED_DURATION_SEC,
    help=MIN_DURATION_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(top_input_dir_name, first_spc_date_string, last_spc_date_string,
         first_time_string, last_time_string, max_velocity_diff_m_s01,
         max_link_distance_m_s01, max_join_time_seconds, max_join_error_m_s01,
         min_duration_seconds, top_output_dir_name):
    """Reanalyzes storm tracks (preferably over many SPC dates).

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param max_velocity_diff_m_s01: Same.
    :param max_link_distance_m_s01: Same.
    :param max_join_time_seconds: Same.
    :param max_join_error_m_s01: Same.
    :param min_duration_seconds: Same.
    :param top_output_dir_name: Same.
    """

    if first_time_string in ['', 'None'] or last_time_string in ['', 'None']:
        first_time_unix_sec = None
        last_time_unix_sec = None
    else:
        first_time_unix_sec = time_conversion.string_to_unix_sec(
            first_time_string, TIME_FORMAT)
        last_time_unix_sec = time_conversion.string_to_unix_sec(
            last_time_string, TIME_FORMAT)

    echo_top_tracking.reanalyze_across_spc_dates(
        top_input_dir_name=top_input_dir_name,
        top_output_dir_name=top_output_dir_name,
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec,
        max_velocity_diff_m_s01=max_velocity_diff_m_s01,
        max_link_distance_m_s01=max_link_distance_m_s01,
        max_join_time_seconds=max_join_time_seconds,
        max_join_error_m_s01=max_join_error_m_s01,
        min_track_duration_seconds=min_duration_seconds)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_SPC_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        max_velocity_diff_m_s01=getattr(
            INPUT_ARG_OBJECT, MAX_VELOCITY_DIFF_ARG_NAME),
        max_link_distance_m_s01=getattr(
            INPUT_ARG_OBJECT, MAX_LINK_DISTANCE_ARG_NAME),
        max_join_time_seconds=getattr(INPUT_ARG_OBJECT, MAX_JOIN_TIME_ARG_NAME),
        max_join_error_m_s01=getattr(INPUT_ARG_OBJECT, MAX_JOIN_ERROR_ARG_NAME),
        min_duration_seconds=getattr(INPUT_ARG_OBJECT, MIN_DURATION_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
