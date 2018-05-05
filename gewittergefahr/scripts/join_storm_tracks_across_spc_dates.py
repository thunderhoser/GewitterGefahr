"""Joins storm tracks across SPC dates.

This gets rid of the cutoff at 1200 UTC of each day.
"""

import argparse
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import echo_top_tracking

# TODO(thunderhoser): All input args to
# `echo_top_tracking.join_tracks_across_spc_dates` should be input args to this
# script.

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
MYRORSS_END_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2012-01-01-000813', TIME_FORMAT)
MYRORSS_START_TIME_UNIX_SEC = 0

FIRST_SPC_DATE_ARG_NAME = 'first_spc_date_string'
LAST_SPC_DATE_ARG_NAME = 'last_spc_date_string'
START_TIME_ARG_NAME = 'start_time_string'
END_TIME_ARG_NAME = 'end_time_string'
RADAR_SOURCE_ARG_NAME = 'radar_source'
ORIG_TRACKING_DIR_ARG_NAME = 'orig_tracking_dir_name'
NEW_TRACKING_DIR_ARG_NAME = 'new_tracking_dir_name'

SPC_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  Tracks will be '
    'joined across all dates from `{0:s}`...`{1:s}`.'
).format(FIRST_SPC_DATE_ARG_NAME, LAST_SPC_DATE_ARG_NAME)
START_TIME_HELP_STRING = (
    'Start time (format "yyyy-mm-dd-HHMMSS").  Must be part of `{0:s}`.'
).format(FIRST_SPC_DATE_ARG_NAME)
END_TIME_HELP_STRING = (
    'End time (format "yyyy-mm-dd-HHMMSS").  Must be part of `{0:s}`.'
).format(LAST_SPC_DATE_ARG_NAME)
RADAR_SOURCE_HELP_STRING = (
    'Source of radar data.  Must be in the following list:\n{0:s}'
).format(str(radar_utils.DATA_SOURCE_IDS))
ORIG_TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with original storm tracks (before joining).')
NEW_TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory for new storm tracks (after joining).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + START_TIME_ARG_NAME, type=str, required=False, default='None',
    help=START_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + END_TIME_ARG_NAME, type=str, required=False, default='None',
    help=END_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_SOURCE_ARG_NAME, type=str, required=False,
    default=radar_utils.MYRORSS_SOURCE_ID, help=RADAR_SOURCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ORIG_TRACKING_DIR_ARG_NAME, type=str, required=True,
    help=ORIG_TRACKING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NEW_TRACKING_DIR_ARG_NAME, type=str, required=True,
    help=NEW_TRACKING_DIR_HELP_STRING)

if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    FIRST_SPC_DATE_STRING = getattr(
        INPUT_ARG_OBJECT, FIRST_SPC_DATE_ARG_NAME)
    LAST_SPC_DATE_STRING = getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_ARG_NAME)
    START_TIME_STRING = getattr(INPUT_ARG_OBJECT, START_TIME_ARG_NAME)
    END_TIME_STRING = getattr(INPUT_ARG_OBJECT, END_TIME_ARG_NAME)
    RADAR_SOURCE_NAME = getattr(INPUT_ARG_OBJECT, RADAR_SOURCE_ARG_NAME)
    TOP_INPUT_DIR_NAME = getattr(INPUT_ARG_OBJECT, ORIG_TRACKING_DIR_ARG_NAME)
    TOP_OUTPUT_DIR_NAME = getattr(INPUT_ARG_OBJECT, NEW_TRACKING_DIR_ARG_NAME)

    if START_TIME_STRING == 'None':
        START_TIME_UNIX_SEC = None
    else:
        START_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
            START_TIME_STRING, TIME_FORMAT)

    if END_TIME_STRING == 'None':
        END_TIME_UNIX_SEC = None
    else:
        END_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
            END_TIME_STRING, TIME_FORMAT)

    if RADAR_SOURCE_NAME == radar_utils.MYRORSS_SOURCE_ID:
        TRACKING_START_TIME_UNIX_SEC = MYRORSS_START_TIME_UNIX_SEC
        TRACKING_END_TIME_UNIX_SEC = MYRORSS_END_TIME_UNIX_SEC
    else:
        TRACKING_START_TIME_UNIX_SEC = None
        TRACKING_END_TIME_UNIX_SEC = None

    echo_top_tracking.join_tracks_across_spc_dates(
        first_spc_date_string=FIRST_SPC_DATE_STRING,
        last_spc_date_string=LAST_SPC_DATE_STRING,
        top_input_dir_name=TOP_INPUT_DIR_NAME,
        top_output_dir_name=TOP_OUTPUT_DIR_NAME,
        tracking_start_time_unix_sec=TRACKING_START_TIME_UNIX_SEC,
        tracking_end_time_unix_sec=TRACKING_END_TIME_UNIX_SEC,
        start_time_unix_sec=START_TIME_UNIX_SEC,
        end_time_unix_sec=END_TIME_UNIX_SEC)
