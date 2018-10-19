"""Shares linkage info across SPC dates.

For more information on this procedure, see documentation for
`link_events_to_storms.share_linkages_across_spc_dates`.
"""

import argparse
from gewittergefahr.gg_utils import link_events_to_storms as events2storms

FIRST_SPC_DATE_ARG_NAME = 'first_spc_date_string'
LAST_SPC_DATE_ARG_NAME = 'last_spc_date_string'
INPUT_DIR_ARG_NAME = 'input_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'
EVENT_TYPE_ARG_NAME = 'event_type_string'

SPC_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  Linkage info '
    'will be shared across all dates from `{0:s}`...`{1:s}`.'
).format(FIRST_SPC_DATE_ARG_NAME, LAST_SPC_DATE_ARG_NAME)

INPUT_DIR_HELP_STRING = (
    'Name of top-level input directory with original linkage files (before '
    'sharing info across SPC dates).')

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory for new linkage files (after sharing '
    'across SPC dates).')

EVENT_TYPE_HELP_STRING = (
    'Type of linkages to be handled (either "wind" or "tornado").')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_ARG_NAME, type=str, required=True,
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

    events2storms.share_linkages_across_spc_dates(
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_SPC_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_ARG_NAME),
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
        event_type_string=getattr(INPUT_ARG_OBJECT, EVENT_TYPE_ARG_NAME))
