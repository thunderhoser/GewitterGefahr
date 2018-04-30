"""Links each storm cell to zero or more tornadoes."""

import argparse
import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import link_events_to_storms as events2storms

# TODO(thunderhoser): All input args to
# `link_events_to_storms.link_each_storm_to_tornadoes` should be input args to
# this script.

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FIRST_SPC_DATE_ARG_NAME = 'first_spc_date_string'
LAST_SPC_DATE_ARG_NAME = 'last_spc_date_string'
TORNADO_DIR_ARG_NAME = 'input_tornado_dir_name'
TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
TRACKING_SCALE_ARG_NAME = 'tracking_scale_metres2'
LINKAGE_DIR_ARG_NAME = 'output_linkage_dir_name'

SPC_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  Each storm cell '
    'on dates `{0:s}`...`{1:s}` will be linked to zero or more tornadoes.'
).format(FIRST_SPC_DATE_ARG_NAME, LAST_SPC_DATE_ARG_NAME)
TORNADO_DIR_HELP_STRING = (
    'Name of top-level directory with tornado observations (files readable by'
    ' `tornado_io.read_processed_file`).')
TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with storm tracks (files readable by'
    ' `storm_tracking_io.read_processed_file`).')
TRACKING_SCALE_HELP_STRING = (
    'Tracking scale (minimum object area).  This is used to find exact files in'
    ' `{0:s}`.').format(TRACKING_DIR_ARG_NAME)
LINKAGE_DIR_HELP_STRING = (
    'Name of top-level directory for linkage files (created by'
    ' `link_events_to_storms.write_storm_to_tornadoes_table`).  One file for '
    'each time step will be created here.')

TOP_TORNADO_DIR_NAME_DEFAULT = (
    '/condo/swatwork/ralager/tornado_observations/processed')
DEFAULT_TRACKING_SCALE_METRES2 = int(numpy.round(
    echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2))

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TORNADO_DIR_ARG_NAME, type=str, required=False,
    default=TOP_TORNADO_DIR_NAME_DEFAULT, help=TORNADO_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_ARG_NAME, type=str, required=True,
    help=TRACKING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_SCALE_ARG_NAME, type=int, required=False,
    default=DEFAULT_TRACKING_SCALE_METRES2, help=TRACKING_SCALE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LINKAGE_DIR_ARG_NAME, type=str, required=True,
    help=LINKAGE_DIR_HELP_STRING)


def _link_tornadoes_to_storms(
        first_spc_date_string, last_spc_date_string, top_tornado_dir_name,
        top_tracking_dir_name, tracking_scale_metres2, top_linkage_dir_name):
    """Links each storm cell to zero or more tornadoes.

    :param first_spc_date_string: See documentation at top of file.
    :param last_spc_date_string: Same.
    :param top_tornado_dir_name: Same.
    :param top_tracking_dir_name: Same.
    :param tracking_scale_metres2: Same.
    :param top_linkage_dir_name: Same.
    """

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    tracking_file_names = []
    num_spc_dates = len(spc_date_strings)

    for i in range(num_spc_dates):
        tracking_file_names += tracking_io.find_processed_files_one_spc_date(
            spc_date_string=spc_date_strings[i],
            data_source=tracking_utils.SEGMOTION_SOURCE_ID,
            top_processed_dir_name=top_tracking_dir_name,
            tracking_scale_metres2=tracking_scale_metres2,
            raise_error_if_missing=True)

    file_times_unix_sec = numpy.array(
        [tracking_io.processed_file_name_to_time(f)
         for f in tracking_file_names], dtype=int)

    storm_to_tornadoes_table = events2storms.link_each_storm_to_tornadoes(
        tracking_file_names=tracking_file_names,
        tornado_directory_name=top_tornado_dir_name)
    print SEPARATOR_STRING

    events2storms.write_linkages_one_file_per_storm_time(
        top_directory_name=top_linkage_dir_name,
        file_times_unix_sec=file_times_unix_sec,
        storm_to_tornadoes_table=storm_to_tornadoes_table)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    FIRST_SPC_DATE_STRING = getattr(INPUT_ARG_OBJECT, FIRST_SPC_DATE_ARG_NAME)
    LAST_SPC_DATE_STRING = getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_ARG_NAME)
    TOP_TORNADO_DIR_NAME = getattr(INPUT_ARG_OBJECT, TORNADO_DIR_ARG_NAME)
    TOP_TRACKING_DIR_NAME = getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME)
    TRACKING_SCALE_METRES2 = getattr(INPUT_ARG_OBJECT, TRACKING_SCALE_ARG_NAME)
    TOP_LINKAGE_DIR_NAME = getattr(INPUT_ARG_OBJECT, LINKAGE_DIR_ARG_NAME)

    _link_tornadoes_to_storms(
        first_spc_date_string=FIRST_SPC_DATE_STRING,
        last_spc_date_string=LAST_SPC_DATE_STRING,
        top_tornado_dir_name=TOP_TORNADO_DIR_NAME,
        top_tracking_dir_name=TOP_TRACKING_DIR_NAME,
        tracking_scale_metres2=TRACKING_SCALE_METRES2,
        top_linkage_dir_name=TOP_LINKAGE_DIR_NAME)
