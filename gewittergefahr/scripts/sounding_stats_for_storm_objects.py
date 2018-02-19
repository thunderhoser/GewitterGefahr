"""Computes sounding statistics for each storm object.

"Sounding statistics" may also be referred to as "sounding indices" or "NSE
(near-storm environment) variables".

--- EXAMPLE USAGE ---

To compute stats at valid time only:

sounding_stats_for_storm_objects.py --spc_date_string="20110404"
--lead_times_seconds 0 --input_ruc_directory_name=${RUC_DIRECTORY_NAME}
--input_tracking_dir_name=${TRACKING_DIR_NAME}
--output_dir_name=${OUTPUT_DIR_NAME}

To compute stats at lead times of 7.5, 22.5, 37.5, 52.5, and 75 minutes:

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
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TIME_FORMAT_IN_OUTPUT_FILE_NAMES = '%Y%m%d%H%M%S'

WGRIB_EXE_NAME = '/condo/swatwork/ralager/wgrib/wgrib'
WGRIB2_EXE_NAME = '/condo/swatwork/ralager/grib2/wgrib2/wgrib2'

SPC_DATE_INPUT_ARG = 'spc_date_string'
START_TIME_INPUT_ARG = 'start_time_unix_sec'
END_TIME_INPUT_ARG = 'end_time_unix_sec'
LEAD_TIMES_INPUT_ARG = 'lead_times_seconds'
TRACKING_SCALE_INPUT_ARG = 'tracking_scale_metres2'
RUC_DIRECTORY_INPUT_ARG = 'input_ruc_directory_name'
TRACKING_DIR_INPUT_ARG = 'input_tracking_dir_name'
OUTPUT_DIR_INPUT_ARG = 'output_dir_name'

SPC_DATE_HELP_STRING = 'See help string for `{0:s}` and `{1:s}`.'.format(
    START_TIME_INPUT_ARG, END_TIME_INPUT_ARG)
TIME_HELP_STRING = (
    'Time in Unix seconds.  Sounding stats will be computed for all storm '
    'objects from `{0:s}`...`{1:s}` on SPC date `{2:s}`.  If `{0:s}` or `{1:s}`'
    ' is left empty, sounding stats will be computed for all storm objects on '
    'SPC date `{2:s}`.').format(
        START_TIME_INPUT_ARG, END_TIME_INPUT_ARG, SPC_DATE_INPUT_ARG)
LEAD_TIMES_HELP_STRING = (
    'Array of lead times.  For each storm object, sounding stats will be '
    'computed for each lead time.')
TRACKING_SCALE_HELP_STRING = (
    'Tracking scale (minimum storm area).  Will be used to find input data.')
RUC_DIRECTORY_HELP_STRING = (
    'Name of top-level directory with Rapid Update Cycle (RUC) data.  RUC data '
    'will be interpolated to storm objects to create soundings.')
TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with storm-tracking data.')
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  A single Pickle file, with sounding stats for '
    'each storm object, will be written here.')

DEFAULT_LEAD_TIMES_SECONDS = [0]
DEFAULT_TRACKING_SCALE_METRES2 = int(numpy.round(
    echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2))

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SPC_DATE_INPUT_ARG, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + START_TIME_INPUT_ARG, type=str, required=False, default=-1,
    help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + END_TIME_INPUT_ARG, type=str, required=False, default=-1,
    help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LEAD_TIMES_INPUT_ARG, type=int, nargs='+', required=False,
    default=[0], help=LEAD_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_SCALE_INPUT_ARG, type=int, required=False,
    default=DEFAULT_TRACKING_SCALE_METRES2, help=TRACKING_SCALE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RUC_DIRECTORY_INPUT_ARG, type=str, required=True,
    help=RUC_DIRECTORY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_INPUT_ARG, type=str, required=True,
    help=TRACKING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_INPUT_ARG, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _get_pathless_output_file_name(
        spc_date_string, default_time_limits, start_time_unix_sec=None,
        end_time_unix_sec=None):
    """Creates pathless name for output file.

    Each output file should contain sounding stats for all storm objects in the
    given time period.

    :param spc_date_string: SPC date for storm objects (format "yyyymmdd").
    :param default_time_limits: Boolean flag.  If default_time_limits = True,
        the file will contain all storm objects in the SPC date.  If
        default_time_limits = False, the file will contain only storm objects in
        the time range `start_time_unix_sec`...`end_time_unix_sec`.
    :param start_time_unix_sec: See documentation for `default_time_limits`.
    :param end_time_unix_sec: See documentation for `default_time_limits`.
    :return: pathless_output_file_name: Pathless name for output file.
    """

    if default_time_limits:
        return 'sounding_statistics_{0:s}.p'.format(spc_date_string)

    return 'sounding_statistics_{0:s}_{1:s}-{2:s}.p'.format(
        spc_date_string,
        time_conversion.unix_sec_to_string(
            start_time_unix_sec, TIME_FORMAT_IN_OUTPUT_FILE_NAMES),
        time_conversion.unix_sec_to_string(
            end_time_unix_sec, TIME_FORMAT_IN_OUTPUT_FILE_NAMES))


def _compute_sounding_stats(
        spc_date_string, start_time_unix_sec, end_time_unix_sec,
        lead_times_seconds, tracking_scale_metres2, top_ruc_directory_name,
        top_tracking_dir_name, output_dir_name):
    """Computes sounding statistics for each storm object.

    :param spc_date_string: See documentation for `start_time_unix_sec`.
    :param start_time_unix_sec: Time in Unix seconds.  Sounding stats will be
        computed for all storm objects from `start_time_unix_sec`...
        `end_time_unix_sec` on SPC date `spc_date_string`.  If
        `start_time_unix_sec` or `end_time_unix_sec` is left empty, sounding
        stats will be computed for all storm objects on SPC date
        `spc_date_string`.
    :param end_time_unix_sec: See documentation for `start_time_unix_sec`.
    :param lead_times_seconds: Array of lead times.  For each storm object,
        sounding stats will be computed for each lead time.
    :param tracking_scale_metres2: Tracking scale (minimum storm area).  Will be
        used to find input data.
    :param top_ruc_directory_name: Name of top-level directory with Rapid Update
        Cycle (RUC) data.  RUC data will be interpolated to storm objects to
        create soundings.
    :param top_tracking_dir_name: Name of top-level directory with storm-
        tracking data.
    :param output_dir_name: Name of output directory.  A single Pickle file,
        with sounding stats for each storm object, will be written here.
    """

    lead_times_seconds = numpy.array(lead_times_seconds, dtype=int)

    if start_time_unix_sec == -1 or end_time_unix_sec == -1:
        start_time_unix_sec = 0
        end_time_unix_sec = int(1e10)
        default_time_limits = True
    else:
        default_time_limits = False

    tracking_file_names = tracking_io.find_processed_files_one_spc_date(
        spc_date_string=spc_date_string,
        data_source=tracking_utils.SEGMOTION_SOURCE_ID,
        top_processed_dir_name=top_tracking_dir_name,
        tracking_scale_metres2=tracking_scale_metres2)

    file_times_unix_sec = [
        tracking_io.processed_file_name_to_time(f) for f in tracking_file_names]
    time_in_range_flags = numpy.array(
        [start_time_unix_sec <= t <= end_time_unix_sec
         for t in file_times_unix_sec])
    time_in_range_indices = numpy.where(time_in_range_flags)[0]

    tracking_file_names = [
        tracking_file_names[i] for i in time_in_range_indices]

    storm_object_table = tracking_io.read_many_processed_files(
        tracking_file_names)
    print SEPARATOR_STRING

    sounding_stat_table = soundings.get_sounding_stats_for_storm_objects(
        storm_object_table=storm_object_table,
        lead_times_seconds=lead_times_seconds, all_ruc_grids=True,
        top_grib_directory_name=top_ruc_directory_name,
        wgrib_exe_name=WGRIB_EXE_NAME, wgrib2_exe_name=WGRIB2_EXE_NAME,
        raise_error_if_missing=False)
    print SEPARATOR_STRING

    num_lead_times = len(lead_times_seconds)
    sounding_stat_file_names = [''] * num_lead_times
    for j in range(num_lead_times):
        this_pathless_file_name = _get_pathless_output_file_name(
            spc_date_string=spc_date_string,
            default_time_limits=default_time_limits,
            start_time_unix_sec=start_time_unix_sec,
            end_time_unix_sec=end_time_unix_sec)

        sounding_stat_file_names[j] = '{0:s}/lead_time_{1:d}sec/{2:s}'.format(
            output_dir_name, lead_times_seconds[j], this_pathless_file_name)

    print 'Writing sounding stats to output files...'
    soundings.write_sounding_stats_for_storm_objects(
        sounding_stat_table, lead_times_seconds=lead_times_seconds,
        pickle_file_names=sounding_stat_file_names)

    print (
        'Sounding stats have been written to the following files:'
        '\n{0:s}').format(sounding_stat_file_names)


def add_input_arguments(argument_parser_object):
    """Adds input args for this script to `argparse.ArgumentParser` object.

    :param argument_parser_object: `argparse.ArgumentParser` object, which may
        or may not already contain input args.
    :return: argument_parser_object: Same as input object, but with new input
        args added.
    """

    argument_parser_object.add_argument(
        '--' + LEAD_TIMES_INPUT_ARG, type=int, nargs='+', required=False,
        default=[0], help=LEAD_TIMES_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + TRACKING_SCALE_INPUT_ARG, type=int, required=False,
        default=DEFAULT_TRACKING_SCALE_METRES2, help=TRACKING_SCALE_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + RUC_DIRECTORY_INPUT_ARG, type=str, required=True,
        help=RUC_DIRECTORY_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + TRACKING_DIR_INPUT_ARG, type=str, required=True,
        help=TRACKING_DIR_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + OUTPUT_DIR_INPUT_ARG, type=str, required=True,
        help=OUTPUT_DIR_HELP_STRING)

    return argument_parser_object


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    SPC_DATE_STRING = getattr(INPUT_ARG_OBJECT, SPC_DATE_INPUT_ARG)
    START_TIME_UNIX_SEC = getattr(INPUT_ARG_OBJECT, START_TIME_INPUT_ARG)
    END_TIME_UNIX_SEC = getattr(INPUT_ARG_OBJECT, END_TIME_INPUT_ARG)
    LEAD_TIMES_SECONDS = getattr(INPUT_ARG_OBJECT, LEAD_TIMES_INPUT_ARG)
    TRACKING_SCALE_METRES2 = getattr(INPUT_ARG_OBJECT, TRACKING_SCALE_INPUT_ARG)
    TOP_RUC_DIRECTORY_NAME = getattr(INPUT_ARG_OBJECT, RUC_DIRECTORY_INPUT_ARG)
    TOP_TRACKING_DIR_NAME = getattr(INPUT_ARG_OBJECT, TRACKING_DIR_INPUT_ARG)
    OUTPUT_DIR_NAME = getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_INPUT_ARG)

    _compute_sounding_stats(
        spc_date_string=SPC_DATE_STRING,
        start_time_unix_sec=START_TIME_UNIX_SEC,
        end_time_unix_sec=END_TIME_UNIX_SEC,
        lead_times_seconds=LEAD_TIMES_SECONDS,
        tracking_scale_metres2=TRACKING_SCALE_METRES2,
        top_ruc_directory_name=TOP_RUC_DIRECTORY_NAME,
        top_tracking_dir_name=TOP_TRACKING_DIR_NAME,
        output_dir_name=OUTPUT_DIR_NAME)
