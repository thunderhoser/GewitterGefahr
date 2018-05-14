"""Computes sounding statistics for each storm object and lead time.

"Sounding statistics" may also be called "sounding indices" or "NSE (near-storm
environment) variables".

--- USAGE EXAMPLES ---

To compute sounding stats only at the time of each storm object (with zero lead
time):

sounding_stats_for_storm_objects.py --spc_date_string="20110404"
--lead_times_seconds 0 --input_ruc_directory_name=${RUC_DIRECTORY_NAME}
--input_tracking_dir_name=${TRACKING_DIR_NAME}
--output_dir_name=${OUTPUT_DIR_NAME}

To compute sounding stats with lead times of 7.5, 22.5, 37.5, 52.5, and 75
minutes:

sounding_stats_for_storm_objects.py --spc_date_string="20110404"
--lead_times_seconds 0 450 1350 2250 3150 4500
--input_ruc_directory_name=${RUC_DIRECTORY_NAME}
--input_tracking_dir_name=${TRACKING_DIR_NAME}
--output_dir_name=${OUTPUT_DIR_NAME}
"""

import argparse
import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

WGRIB_EXE_NAME = '/condo/swatwork/ralager/wgrib/wgrib'
WGRIB2_EXE_NAME = '/condo/swatwork/ralager/grib2/wgrib2/wgrib2'

SPC_DATE_ARG_NAME = 'spc_date_string'
LEAD_TIMES_ARG_NAME = 'lead_times_seconds'
TRACKING_SCALE_ARG_NAME = 'tracking_scale_metres2'
RUC_DIRECTORY_ARG_NAME = 'input_ruc_directory_name'
TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

SPC_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  Sounding '
    'statistics will be computed for all storm objects on this date at all lead'
    ' times in `{0:s}`.').format(LEAD_TIMES_ARG_NAME)
LEAD_TIMES_HELP_STRING = (
    'Array of lead times.  Sounding statistics will be computed for all storm '
    'objects on date `{0:s}` at all lead times.').format(SPC_DATE_ARG_NAME)
TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with storm-tracking data (one file per time '
    'step, readable by `storm_tracking_io.read_processed_file`).')
TRACKING_SCALE_HELP_STRING = (
    'Tracking scale (minimum storm area).  Used to find input data.')
RUC_DIRECTORY_HELP_STRING = (
    'Name of top-level directory with grib files containing RUC (Rapid Update '
    'Cycle) data.  These data will be interpolated to storm objects to create '
    'soundings.')
OUTPUT_DIR_HELP_STRING = (
    'Name of top-level directory for output files.  There will be one file for '
    'each storm time and lead time, written by '
    '`soundings.write_sounding_statistics`.')

DEFAULT_LEAD_TIMES_SECONDS = [0]
DEFAULT_TRACKING_SCALE_METRES2 = int(numpy.round(
    echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2))

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LEAD_TIMES_ARG_NAME, type=int, nargs='+', required=False,
    default=[0], help=LEAD_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_ARG_NAME, type=str, required=True,
    help=TRACKING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_SCALE_ARG_NAME, type=int, required=False,
    default=DEFAULT_TRACKING_SCALE_METRES2, help=TRACKING_SCALE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RUC_DIRECTORY_ARG_NAME, type=str, required=True,
    help=RUC_DIRECTORY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _compute_statistics(
        spc_date_string, lead_times_seconds, top_tracking_dir_name,
        tracking_scale_metres2, top_ruc_directory_name, top_output_dir_name):
    """Computes sounding statistics for each storm object and lead time.

    :param spc_date_string: See documentation at top of file.
    :param lead_times_seconds: Same.
    :param top_tracking_dir_name: Same.
    :param tracking_scale_metres2: Same.
    :param top_ruc_directory_name: Same.
    :param top_output_dir_name: Same.
    """

    lead_times_seconds = numpy.array(lead_times_seconds, dtype=int)

    tracking_file_names = tracking_io.find_processed_files_one_spc_date(
        spc_date_string=spc_date_string,
        data_source=tracking_utils.SEGMOTION_SOURCE_ID,
        top_processed_dir_name=top_tracking_dir_name,
        tracking_scale_metres2=tracking_scale_metres2)

    storm_object_table = tracking_io.read_many_processed_files(
        tracking_file_names)
    print SEPARATOR_STRING

    sounding_statistic_table = soundings.get_sounding_stats_for_storm_objects(
        storm_object_table=storm_object_table,
        top_grib_directory_name=top_ruc_directory_name,
        lead_times_sec=lead_times_seconds, all_ruc_grids=True,
        wgrib_exe_name=WGRIB_EXE_NAME, wgrib2_exe_name=WGRIB2_EXE_NAME,
        raise_error_if_missing=False)
    print SEPARATOR_STRING

    soundings.write_sounding_statistics_many_times(
        top_directory_name=top_output_dir_name,
        sounding_statistic_table=sounding_statistic_table)


def add_input_arguments(argument_parser_object):
    """Adds input args for this script to `argparse.ArgumentParser` object.

    :param argument_parser_object: `argparse.ArgumentParser` object, which may
        or may not already contain input args.
    :return: argument_parser_object: Same as input object, but with new input
        args added.
    """

    argument_parser_object.add_argument(
        '--' + LEAD_TIMES_ARG_NAME, type=int, nargs='+', required=False,
        default=[0], help=LEAD_TIMES_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + TRACKING_SCALE_ARG_NAME, type=int, required=False,
        default=DEFAULT_TRACKING_SCALE_METRES2, help=TRACKING_SCALE_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + RUC_DIRECTORY_ARG_NAME, type=str, required=True,
        help=RUC_DIRECTORY_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + TRACKING_DIR_ARG_NAME, type=str, required=True,
        help=TRACKING_DIR_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
        help=OUTPUT_DIR_HELP_STRING)

    return argument_parser_object


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    SPC_DATE_STRING = getattr(INPUT_ARG_OBJECT, SPC_DATE_ARG_NAME)
    LEAD_TIMES_SECONDS = getattr(INPUT_ARG_OBJECT, LEAD_TIMES_ARG_NAME)
    TRACKING_SCALE_METRES2 = getattr(INPUT_ARG_OBJECT, TRACKING_SCALE_ARG_NAME)
    TOP_RUC_DIRECTORY_NAME = getattr(INPUT_ARG_OBJECT, RUC_DIRECTORY_ARG_NAME)
    TOP_TRACKING_DIR_NAME = getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME)
    TOP_OUTPUT_DIR_NAME = getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)

    _compute_statistics(
        spc_date_string=SPC_DATE_STRING,
        lead_times_seconds=LEAD_TIMES_SECONDS,
        tracking_scale_metres2=TRACKING_SCALE_METRES2,
        top_ruc_directory_name=TOP_RUC_DIRECTORY_NAME,
        top_tracking_dir_name=TOP_TRACKING_DIR_NAME,
        top_output_dir_name=TOP_OUTPUT_DIR_NAME)
