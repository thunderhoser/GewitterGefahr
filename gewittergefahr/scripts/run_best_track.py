"""Runs the best-track algorithm with default parameters.

Best-track corrects storm tracks produced by segmotion.

--- DEFINITIONS ---

segmotion = a storm-tracking algorithm in WDSS-II (Lakshmanan and Smith 2010)

WDSS-II = the Warning Decision Support System with Integrated Information
(Lakshmanan et. al 2007)

--- REFERENCES ---

Lakshmanan, V., T. Smith, G. Stumpf, and K. Hondl, 2007: "The Warning Decision
    Support System -- Integrated Information". Weather and Forecasting, 22 (3),
    596-612.

Lakshmanan, V., and T. Smith, 2010: "Evaluating a storm tracking algorithm".
    26th Conference on Interactive Information Processing Systems, Atlanta, GA,
    American Meteorological Society.
"""

import os
import argparse
from gewittergefahr.gg_utils import best_tracks
from gewittergefahr.gg_utils import best_tracks_smart_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import time_conversion

# TODO(thunderhoser): All input args to `best_tracks.run_best_track` should also
# be input args to this script.

# TODO(thunderhoser): Add option to use best_tracks_smart_io.py for long time
# periods.

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
DEFAULT_TRACKING_SCALE_METRES = int(4e7)

FIRST_SPC_DATE_INPUT_ARG = 'first_spc_date_string'
LAST_SPC_DATE_INPUT_ARG = 'last_spc_date_string'
SEGMOTION_DIR_INPUT_ARG = 'input_segmotion_dir_name'
BEST_TRACK_DIR_INPUT_ARG = 'output_best_track_dir_name'
TRACKING_SCALE_INPUT_ARG = 'tracking_scale_metres2'

SPC_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  Best-track will '
    'be run for all dates from `{0:s}`...`{1:s}`.').format(
        FIRST_SPC_DATE_INPUT_ARG, LAST_SPC_DATE_INPUT_ARG)
SEGMOTION_DIR_HELP_STRING = (
    'Name of top-level directory with segmotion tracks (which will be corrected'
    ' by best-track).')
BEST_TRACK_DIR_HELP_STRING = (
    'Name of top-level directory for best-track results.')
TRACKING_SCALE_HELP_STRING = (
    'Tracking scale (minimum storm area) for segmotion.  Used only to find '
    'input files.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_INPUT_ARG, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_INPUT_ARG, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SEGMOTION_DIR_INPUT_ARG, type=str, required=True,
    help=SEGMOTION_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + BEST_TRACK_DIR_INPUT_ARG, type=str, required=True,
    help=BEST_TRACK_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_SCALE_INPUT_ARG, type=int, required=False,
    default=DEFAULT_TRACKING_SCALE_METRES, help=TRACKING_SCALE_HELP_STRING)


def _create_best_tracks(
        first_spc_date_string, last_spc_date_string, top_segmotion_dir_name,
        top_best_track_dir_name, tracking_scale_metres2):
    """Runs the best-track algorithm with default parameters.

    :param first_spc_date_string: SPC (Storm Prediction Center) date in format
        "yyyymmdd".  Best-track will be run for all dates from
        `first_spc_date_string`...`last_spc_date_string`.
    :param last_spc_date_string: See above.
    :param top_segmotion_dir_name: [input] Name of top-level directory with
        segmotion tracks (which will be corrected by best-track)
    :param top_best_track_dir_name: [output] Name of top-level directory for
        best-track results.
    :param tracking_scale_metres2: Tracking scale (minimum storm area) for
        segmotion.  Used only to find input files.
    """

    start_time_unix_sec = time_conversion.MIN_SECONDS_INTO_SPC_DATE + (
        time_conversion.string_to_unix_sec(
            first_spc_date_string, time_conversion.SPC_DATE_FORMAT))
    end_time_unix_sec = time_conversion.MAX_SECONDS_INTO_SPC_DATE + (
        time_conversion.string_to_unix_sec(
            last_spc_date_string, time_conversion.SPC_DATE_FORMAT))

    file_dictionary = best_tracks_smart_io.find_files_for_smart_io(
        start_time_unix_sec=start_time_unix_sec,
        start_spc_date_string=first_spc_date_string,
        end_time_unix_sec=end_time_unix_sec,
        end_spc_date_string=last_spc_date_string,
        data_source=tracking_utils.SEGMOTION_SOURCE_ID,
        tracking_scale_metres2=tracking_scale_metres2,
        top_input_dir_name=top_segmotion_dir_name,
        top_output_dir_name=top_best_track_dir_name)

    num_spc_dates = len(file_dictionary[best_tracks_smart_io.SPC_DATES_KEY])
    input_file_names = []
    output_file_names = []

    for i in range(num_spc_dates):
        input_file_names += file_dictionary[
            best_tracks_smart_io.INPUT_FILE_NAMES_KEY][i]
        output_file_names += file_dictionary[
            best_tracks_smart_io.OUTPUT_FILE_NAMES_KEY][i]

    temp_file_names = file_dictionary[best_tracks_smart_io.TEMP_FILE_NAMES_KEY]
    for this_temp_file_name in temp_file_names:
        os.remove(this_temp_file_name)

    storm_object_table = best_tracks.read_input_storm_objects(input_file_names)
    print SEPARATOR_STRING

    storm_object_table = best_tracks.run_best_track(
        storm_object_table=storm_object_table)
    print SEPARATOR_STRING

    best_tracks.write_output_storm_objects(
        storm_object_table, input_file_names=input_file_names,
        output_file_names=output_file_names)


def add_input_arguments(argument_parser_object):
    """Adds input args for this script to `argparse.ArgumentParser` object.

    :param argument_parser_object: `argparse.ArgumentParser` object, which may
        or may not already contain input args.
    :return: argument_parser_object: Same as input object, but with additional
        input args added.
    """

    argument_parser_object.add_argument(
        '--' + FIRST_SPC_DATE_INPUT_ARG, type=str, required=True,
        help=SPC_DATE_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + LAST_SPC_DATE_INPUT_ARG, type=str, required=True,
        help=SPC_DATE_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + SEGMOTION_DIR_INPUT_ARG, type=str, required=True,
        help=SEGMOTION_DIR_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + BEST_TRACK_DIR_INPUT_ARG, type=str, required=True,
        help=BEST_TRACK_DIR_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + TRACKING_SCALE_INPUT_ARG, type=int, required=False,
        default=DEFAULT_TRACKING_SCALE_METRES, help=TRACKING_SCALE_HELP_STRING)

    return argument_parser_object


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    FIRST_SPC_DATE_STRING = getattr(
        INPUT_ARG_OBJECT, FIRST_SPC_DATE_INPUT_ARG)
    LAST_SPC_DATE_STRING = getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_INPUT_ARG)
    TOP_SEGMOTION_DIR_NAME = getattr(INPUT_ARG_OBJECT, SEGMOTION_DIR_INPUT_ARG)
    TOP_BEST_TRACK_DIR_NAME = getattr(
        INPUT_ARG_OBJECT, BEST_TRACK_DIR_INPUT_ARG)
    TRACKING_SCALE_METRES2 = getattr(INPUT_ARG_OBJECT, TRACKING_SCALE_INPUT_ARG)

    _create_best_tracks(
        first_spc_date_string=FIRST_SPC_DATE_STRING,
        last_spc_date_string=LAST_SPC_DATE_STRING,
        top_segmotion_dir_name=TOP_SEGMOTION_DIR_NAME,
        top_best_track_dir_name=TOP_BEST_TRACK_DIR_NAME,
        tracking_scale_metres2=TRACKING_SCALE_METRES2)
