"""Runs `linkage.link_winds_to_storms`."""

import argparse
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import linkage

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

WIND_DIR_ARG_NAME = 'input_wind_dir_name'
TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
TRACKING_SCALE_ARG_NAME = 'tracking_scale_metres2'
FIRST_SPC_DATE_ARG_NAME = 'first_spc_date_string'
LAST_SPC_DATE_ARG_NAME = 'last_spc_date_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

WIND_DIR_HELP_STRING = (
    'Name of top-level wind directory.  Files therein will be found by '
    '`raw_wind_io.find_processed_hourly_files` and read by '
    '`raw_wind_io.read_processed_file`.')

TRACKING_DIR_HELP_STRING = (
    'Name of top-level tracking directory.  Files therein will be found by '
    '`storm_tracking_io.find_processed_files_one_spc_date` and read by '
    '`storm_tracking_io.read_processed_file`.')

TRACKING_SCALE_HELP_STRING = (
    'Tracking scale (minimum object area).  Used to find files in `{0:s}`.'
).format(TRACKING_DIR_ARG_NAME)

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  This script will operate independently on'
    ' each set of consecutive days in `{0:s}`...`{1:s}`.'
).format(FIRST_SPC_DATE_ARG_NAME, LAST_SPC_DATE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level directory for linkage files.  Files will be written by '
    '`linkage.write_linkage_file`, to locations therein determined by '
    '`linkage.find_linkage_file`.')

TOP_WIND_DIR_NAME_DEFAULT = (
    '/condo/swatwork/ralager/wind_observations/processed')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + WIND_DIR_ARG_NAME, type=str, required=False,
    default=TOP_WIND_DIR_NAME_DEFAULT, help=WIND_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_ARG_NAME, type=str, required=True,
    help=TRACKING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_SCALE_ARG_NAME, type=int, required=False,
    default=echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2,
    help=TRACKING_SCALE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _link_winds_one_period(
        tracking_file_names, top_wind_dir_name, top_output_dir_name):
    """Links tornadoes to storms for one continuous period.

    :param tracking_file_names: 1-D list of paths to tracking files.  Each will
        be read by `storm_tracking_io.read_processed_file`.
    :param top_wind_dir_name: See documentation at top of file.
    :param top_output_dir_name: Same.
    """

    if len(tracking_file_names) == 0:
        return

    try:
        storm_to_winds_table = linkage.link_storms_to_winds(
            tracking_file_names=tracking_file_names,
            top_wind_directory_name=top_wind_dir_name)
        print SEPARATOR_STRING
    except:
        print SEPARATOR_STRING
        return

    spc_date_string = time_conversion.time_to_spc_date_string(
        tracking_io.file_name_to_time(tracking_file_names[0])
    )

    output_file_name = linkage.find_linkage_file(
        top_directory_name=top_output_dir_name,
        event_type_string=linkage.WIND_EVENT_STRING,
        spc_date_string=spc_date_string, raise_error_if_missing=False)

    print 'Writing linkages to: "{0:s}"...'.format(output_file_name)
    linkage.write_linkage_file(
        storm_to_events_table=storm_to_winds_table,
        pickle_file_name=output_file_name)
    print SEPARATOR_STRING


def _run(top_wind_dir_name, top_tracking_dir_name, tracking_scale_metres2,
         first_spc_date_string, last_spc_date_string, top_output_dir_name):
    """Runs `linkage.link_winds_to_storms`.

    This is effectively the main method.

    :param top_wind_dir_name: See documentation at top of file.
    :param top_tracking_dir_name: Same.
    :param tracking_scale_metres2: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param top_output_dir_name: Same.
    """

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    tracking_file_names = []

    for this_spc_date_string in spc_date_strings:
        these_file_names = tracking_io.find_files_one_spc_date(
            top_tracking_dir_name=top_tracking_dir_name,
            tracking_scale_metres2=tracking_scale_metres2,
            source_name=tracking_utils.SEGMOTION_NAME,
            spc_date_string=this_spc_date_string, raise_error_if_missing=False
        )[0]

        if len(these_file_names) == 0:
            _link_winds_one_period(
                tracking_file_names=tracking_file_names,
                top_wind_dir_name=top_wind_dir_name,
                top_output_dir_name=top_output_dir_name)

            print SEPARATOR_STRING

            tracking_file_names = []
            continue

        tracking_file_names += these_file_names

    _link_winds_one_period(
        tracking_file_names=tracking_file_names,
        top_wind_dir_name=top_wind_dir_name,
        top_output_dir_name=top_output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_wind_dir_name=getattr(INPUT_ARG_OBJECT, WIND_DIR_ARG_NAME),
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME),
        tracking_scale_metres2=getattr(
            INPUT_ARG_OBJECT, TRACKING_SCALE_ARG_NAME),
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_SPC_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
