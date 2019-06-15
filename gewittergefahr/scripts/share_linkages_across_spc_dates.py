"""Runs `linkage.share_linkages_across_spc_dates`."""

import argparse
from gewittergefahr.gg_utils import linkage

FIRST_DATE_ARG_NAME = 'first_spc_date_string'
LAST_DATE_ARG_NAME = 'last_spc_date_string'
INPUT_DIR_ARG_NAME = 'input_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'
EVENT_TYPE_ARG_NAME = 'event_type_string'

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Linkages will be shared across all dates '
    'from `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

INPUT_DIR_HELP_STRING = (
    'Name of top-level input directory.  Files therein will be found by '
    '`find_linkage_file` and read by `read_linkage_file`.')

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Files will be written by '
    '`write_linkage_file` to locations therein, determined by '
    '`find_linkage_file`.')

EVENT_TYPE_HELP_STRING = (
    'Event type (must be accepted by `linkage.check_event_type`).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EVENT_TYPE_ARG_NAME, type=str, required=True,
    help=EVENT_TYPE_HELP_STRING)

if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    linkage.share_linkages(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        event_type_string=getattr(INPUT_ARG_OBJECT, EVENT_TYPE_ARG_NAME)
    )
