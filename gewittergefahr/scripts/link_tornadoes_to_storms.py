"""Runs `link_events_to_storms.link_each_storm_to_tornadoes` w/ default params.
"""

import argparse
import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import link_events_to_storms as events2storms
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods

# TODO(thunderhoser): All input args to
# `link_events_to_storms.link_each_storm_to_tornadoes` should also be input args
# to this script.

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SPC_DATE_FORMAT = '%Y%m%d'
DAYS_TO_SECONDS = 86400

TRACKING_SCALE_METRES2 = int(numpy.round(
    echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2))

FIRST_SPC_DATE_INPUT_ARG = 'first_spc_date_string'
LAST_SPC_DATE_INPUT_ARG = 'last_spc_date_string'
TRACKING_DIR_INPUT_ARG = 'input_tracking_dir_name'
TORNADO_DIR_INPUT_ARG = 'input_tornado_dir_name'
LINKAGE_FILE_INPUT_ARG = 'output_linkage_file_name'

SPC_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  Linkage will be'
    ' run for all dates from `{0:s}`...`{1:s}`.').format(
        FIRST_SPC_DATE_INPUT_ARG, LAST_SPC_DATE_INPUT_ARG)
TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with storm-tracking files (readable by '
    '`storm_tracking_io.read_processed_file`).')
TORNADO_DIR_HELP_STRING = (
    'Name of directory with tornado files (readable by '
    '`tornado_io.read_processed_file`).')
LINKAGE_FILE_HELP_STRING = (
    'Name of output file (to be written by '
    '`link_events_to_storms.write_storm_to_tornadoes_table`).')

DEFAULT_TRACKING_DIR_NAME = '/condo/swatcommon/common/gridrad/echo_top_tracking'
DEFAULT_TORNADO_DIR_NAME = '/condo/swatwork/ralager/tornado_observations'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_INPUT_ARG, type=str, required=True,
    help=SPC_DATE_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_INPUT_ARG, type=str, required=True,
    help=SPC_DATE_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_INPUT_ARG, type=str, required=False,
    default=DEFAULT_TRACKING_DIR_NAME, help=TRACKING_DIR_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + TORNADO_DIR_INPUT_ARG, type=str, required=False,
    default=DEFAULT_TORNADO_DIR_NAME, help=TORNADO_DIR_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + LINKAGE_FILE_INPUT_ARG, type=str, required=True,
    help=LINKAGE_FILE_HELP_STRING)


def _link_tornadoes_to_storms(
        first_spc_date_string, last_spc_date_string, top_tracking_dir_name,
        top_tornado_dir_name, linkage_file_name):
    """Runs `link_events_to_storms.link_each_storm_to_tornadoes` with defaults.

    :param first_spc_date_string: SPC (Storm Prediction Center) date in format
        "yyyymmdd".  Linkage will be run for all dates from
        `first_spc_date_string`...`last_spc_date_string`.
    :param last_spc_date_string: See above.
    :param top_tracking_dir_name: Name of top-level directory with storm-
        tracking files (readable by `storm_tracking_io.read_processed_file`).
    :param top_tornado_dir_name: Name of directory with tornado files (readable
        by `tornado_io.read_processed_file`).
    :param linkage_file_name: Name of output file (to be written by
        `link_events_to_storms.write_storm_to_tornadoes_table`).
    """

    first_spc_date_unix_sec = time_conversion.string_to_unix_sec(
        first_spc_date_string, SPC_DATE_FORMAT)
    last_spc_date_unix_sec = time_conversion.string_to_unix_sec(
        last_spc_date_string, SPC_DATE_FORMAT)

    spc_dates_unix_sec = time_conversion.SECONDS_INTO_SPC_DATE_DEFAULT + (
        time_periods.range_and_interval_to_list(
            start_time_unix_sec=first_spc_date_unix_sec,
            end_time_unix_sec=last_spc_date_unix_sec,
            time_interval_sec=DAYS_TO_SECONDS, include_endpoint=True))
    spc_date_strings = [
        time_conversion.time_to_spc_date_string(t) for t in spc_dates_unix_sec]

    tracking_file_names = []
    for this_spc_date_string in spc_date_strings:
        tracking_file_names += tracking_io.find_processed_files_one_spc_date(
            spc_date_string=this_spc_date_string,
            data_source=tracking_utils.SEGMOTION_SOURCE_ID,
            top_processed_dir_name=top_tracking_dir_name,
            tracking_scale_metres2=TRACKING_SCALE_METRES2)

    tracking_file_names.sort()
    storm_to_tornadoes_table = events2storms.link_each_storm_to_tornadoes(
        tracking_file_names=tracking_file_names,
        tornado_directory_name=top_tornado_dir_name)
    print SEPARATOR_STRING

    print 'Writing linkages to file: "{0:s}"...'.format(linkage_file_name)
    events2storms.write_storm_to_tornadoes_table(
        storm_to_tornadoes_table, linkage_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    FIRST_SPC_DATE_STRING = getattr(
        INPUT_ARG_OBJECT, FIRST_SPC_DATE_INPUT_ARG)
    LAST_SPC_DATE_STRING = getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_INPUT_ARG)
    TOP_TRACKING_DIR_NAME = getattr(INPUT_ARG_OBJECT, TRACKING_DIR_INPUT_ARG)
    TOP_TORNADO_DIR_NAME = getattr(INPUT_ARG_OBJECT, TORNADO_DIR_INPUT_ARG)
    LINKAGE_FILE_NAME = getattr(INPUT_ARG_OBJECT, LINKAGE_FILE_INPUT_ARG)

    _link_tornadoes_to_storms(
        first_spc_date_string=FIRST_SPC_DATE_STRING,
        last_spc_date_string=LAST_SPC_DATE_STRING,
        top_tracking_dir_name=TOP_TRACKING_DIR_NAME,
        top_tornado_dir_name=TOP_TORNADO_DIR_NAME,
        linkage_file_name=LINKAGE_FILE_NAME)
