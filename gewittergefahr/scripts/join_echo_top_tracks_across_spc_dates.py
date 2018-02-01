"""Joins echo-top-based storm tracks across SPC dates.

This gets rid of the arbitrary cutoff at 1200 UTC of each SPC date.  For more
details, see `echo_top_tracking.join_tracks_across_spc_dates`.
"""

import argparse
from gewittergefahr.gg_utils import echo_top_tracking

# TODO(thunderhoser): All input args to
# `echo_top_tracking.join_tracks_across_spc_dates` should be input args to this
# script.

FIRST_SPC_DATE_INPUT_ARG = 'first_spc_date_string'
LAST_SPC_DATE_INPUT_ARG = 'last_spc_date_string'
ORIG_TRACKING_DIR_INPUT_ARG = 'orig_tracking_dir_name'
NEW_TRACKING_DIR_INPUT_ARG = 'new_tracking_dir_name'

SPC_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  Tracks will be '
    'joined across all dates from `{0:s}`...`{1:s}`.').format(
        FIRST_SPC_DATE_INPUT_ARG, LAST_SPC_DATE_INPUT_ARG)
ORIG_TRACKING_DIR_HELP_STRING = (
    '[input] Name of top-level directory with original storm tracks (before '
    'joining).')
NEW_TRACKING_DIR_HELP_STRING = (
    '[output] Name of top-level directory for new storm tracks (after '
    'joining).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_INPUT_ARG, type=str, required=True,
    help=SPC_DATE_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_INPUT_ARG, type=str, required=True,
    help=SPC_DATE_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + ORIG_TRACKING_DIR_INPUT_ARG, type=str, required=True,
    help=ORIG_TRACKING_DIR_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + NEW_TRACKING_DIR_INPUT_ARG, type=str, required=True,
    help=NEW_TRACKING_DIR_HELP_STRING)

if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    FIRST_SPC_DATE_STRING = getattr(
        INPUT_ARG_OBJECT, FIRST_SPC_DATE_INPUT_ARG)
    LAST_SPC_DATE_STRING = getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_INPUT_ARG)
    TOP_INPUT_DIR_NAME = getattr(INPUT_ARG_OBJECT, ORIG_TRACKING_DIR_INPUT_ARG)
    TOP_OUTPUT_DIR_NAME = getattr(INPUT_ARG_OBJECT, NEW_TRACKING_DIR_INPUT_ARG)

    echo_top_tracking.join_tracks_across_spc_dates(
        first_spc_date_string=FIRST_SPC_DATE_STRING,
        last_spc_date_string=LAST_SPC_DATE_STRING,
        top_input_dir_name=TOP_INPUT_DIR_NAME,
        top_output_dir_name=TOP_OUTPUT_DIR_NAME)
