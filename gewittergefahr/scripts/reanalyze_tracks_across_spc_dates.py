"""Reanalyzes storm tracks across many SPC dates."""

import argparse
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import echo_top_tracking

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
MYRORSS_START_TIME_STRING = '1990-01-01-000000'
MYRORSS_END_TIME_STRING = '2012-01-01-000813'

INPUT_DIR_ARG_NAME = 'input_tracking_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_tracking_dir_name'
FIRST_SPC_DATE_ARG_NAME = 'first_spc_date_string'
LAST_SPC_DATE_ARG_NAME = 'last_spc_date_string'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
RADAR_SOURCE_ARG_NAME = 'radar_source_name'
FOR_STORM_CLIMO_ARG_NAME = 'for_storm_climatology'

INPUT_DIR_HELP_STRING = (
    'Name of top-level input directory.  Files (containing original tracks) '
    'will be found therein by `echo_top_tracking._find_input_tracking_files` '
    'and read by `storm_tracking_io.read_processed_file`.')

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Files (containing reanalyzed tracks) '
    'will be written by `storm_tracking_io.write_processed_file`, to locations '
    'therein determined by `storm_tracking_io.find_processed_file`.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Reanalysis will be done for the period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_SPC_DATE_ARG_NAME, LAST_SPC_DATE_ARG_NAME)

FIRST_TIME_HELP_STRING = (
    'First time in period (format "yyyy-mm-dd-HHMMSS").  Default is 120000 UTC '
    'at beginning of first SPC date.')

LAST_TIME_HELP_STRING = (
    'Last time in period (format "yyyy-mm-dd-HHMMSS").  Default is 115959 UTC '
    'at end of last SPC date.')

RADAR_SOURCE_HELP_STRING = (
    'Source of radar data (must be accepted by '
    '`radar_utils.check_data_source`).')

FOR_STORM_CLIMO_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Boolean flag.  If 1, tracks created by this'
    ' script will be used to create a climatology, in which case start/end '
    'times of tracking period will be set to {2:s} and {3:s}.  Otherwise, '
    'start/end times of tracking period will be set to first and last times '
    'handled by this script.'
).format(RADAR_SOURCE_ARG_NAME, radar_utils.MYRORSS_SOURCE_ID,
         MYRORSS_START_TIME_STRING, MYRORSS_END_TIME_STRING)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)

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
    '--' + RADAR_SOURCE_ARG_NAME, type=str, required=True,
    help=RADAR_SOURCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FOR_STORM_CLIMO_ARG_NAME, type=int, required=False, default=0,
    help=FOR_STORM_CLIMO_HELP_STRING)


def _run(top_input_dir_name, top_output_dir_name, first_spc_date_string,
         last_spc_date_string, first_time_string, last_time_string,
         radar_source_name, for_storm_climatology):
    """Reanalyzes storm tracks across many SPC dates.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param top_output_dir_name: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param radar_source_name: Same.
    :param for_storm_climatology: Same.
    """

    if (for_storm_climatology and
            radar_source_name == radar_utils.MYRORSS_SOURCE_ID):

        myrorss_start_time_unix_sec = time_conversion.string_to_unix_sec(
            MYRORSS_START_TIME_STRING, TIME_FORMAT)
        myrorss_end_time_unix_sec = time_conversion.string_to_unix_sec(
            MYRORSS_END_TIME_STRING, TIME_FORMAT)

        tracking_start_time_unix_sec = myrorss_start_time_unix_sec + 0
        tracking_end_time_unix_sec = myrorss_end_time_unix_sec + 0
    else:
        tracking_start_time_unix_sec = None
        tracking_end_time_unix_sec = None

    if first_time_string in ['', 'None'] or last_time_string in ['', 'None']:
        first_time_unix_sec = None
        last_time_unix_sec = None
    else:
        first_time_unix_sec = time_conversion.string_to_unix_sec(
            first_time_string, TIME_FORMAT)
        last_time_unix_sec = time_conversion.string_to_unix_sec(
            last_time_string, TIME_FORMAT)

    echo_top_tracking.reanalyze_tracks_across_spc_dates(
        top_input_dir_name=top_input_dir_name,
        top_output_dir_name=top_output_dir_name,
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec,
        tracking_start_time_unix_sec=tracking_start_time_unix_sec,
        tracking_end_time_unix_sec=tracking_end_time_unix_sec,
        min_track_duration_seconds=0)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_SPC_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        radar_source_name=getattr(INPUT_ARG_OBJECT, RADAR_SOURCE_ARG_NAME),
        for_storm_climatology=bool(getattr(
            INPUT_ARG_OBJECT, FOR_STORM_CLIMO_ARG_NAME))
    )
