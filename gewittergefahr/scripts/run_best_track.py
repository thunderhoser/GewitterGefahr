"""Runs the best-track algorithm with default parameters.

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

INPUT_TIME_FORMAT = '%Y-%m-%d-%H%M%S'

START_TIME_ARG_NAME = 'start_time_string'
END_TIME_ARG_NAME = 'end_time_string'
INPUT_DIR_ARG_NAME = 'input_tracking_dir_name'
DATA_SOURCE_ARG_NAME = 'data_source'
OUTPUT_DIR_ARG_NAME = 'output_tracking_dir_name'
TRACKING_SCALE_ARG_NAME = 'tracking_scale_metres2'

TIME_HELP_STRING = (
    'Time in format "yyyy-mm-dd-HHMMSS".  Best-track will be run for all time '
    'steps from `{0:s}`...`{1:s}`.'
).format(START_TIME_ARG_NAME, END_TIME_ARG_NAME)
INPUT_DIR_HELP_STRING = (
    'Name of top-level directory with input tracks (files readable by '
    '`storm_tracking_io.read_processed_file`), which will be corrected by '
    'best-track.')
DATA_SOURCE_HELP_STRING = (
    'Data source (must be accepted by '
    '`storm_tracking_utils.check_data_source`).')
OUTPUT_DIR_HELP_STRING = (
    'Name of top-level directory for output tracks (files to be written by '
    '`storm_tracking_io.write_processed_file`), after best-track correction.')
TRACKING_SCALE_HELP_STRING = (
    'Tracking scale (minimum storm area).  Used only to find input files.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + START_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + END_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DATA_SOURCE_ARG_NAME, type=str, required=True,
    help=DATA_SOURCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_SCALE_ARG_NAME, type=int, required=True,
    help=TRACKING_SCALE_HELP_STRING)


def _create_best_tracks(
        start_time_string, end_time_string, top_input_dir_name, data_source,
        top_output_dir_name, tracking_scale_metres2):
    """Runs the best-track algorithm with default parameters.

    :param start_time_string: See documentation at top of file.
    :param end_time_string: Same.
    :param top_input_dir_name: Same.
    :param data_source: Same.
    :param top_output_dir_name: Same.
    :param tracking_scale_metres2: Same.
    """

    tracking_utils.check_data_source(data_source)

    start_time_unix_sec = time_conversion.string_to_unix_sec(
        start_time_string, INPUT_TIME_FORMAT)
    end_time_unix_sec = time_conversion.string_to_unix_sec(
        end_time_string, INPUT_TIME_FORMAT)

    first_spc_date_string = time_conversion.time_to_spc_date_string(
        start_time_unix_sec)
    last_spc_date_string = time_conversion.time_to_spc_date_string(
        end_time_unix_sec)

    file_dictionary = best_tracks_smart_io.find_files_for_smart_io(
        start_time_unix_sec=start_time_unix_sec,
        start_spc_date_string=first_spc_date_string,
        end_time_unix_sec=end_time_unix_sec,
        end_spc_date_string=last_spc_date_string, data_source=data_source,
        tracking_scale_metres2=tracking_scale_metres2,
        top_input_dir_name=top_input_dir_name,
        top_output_dir_name=top_output_dir_name)

    best_tracks_smart_io.run_best_track(smart_file_dict=file_dictionary)


def add_input_arguments(argument_parser_object):
    """Adds input args for this script to `argparse.ArgumentParser` object.

    :param argument_parser_object: `argparse.ArgumentParser` object, which may
        or may not already contain input args.
    :return: argument_parser_object: Same as input object, but with new input
        args added.
    """

    argument_parser_object.add_argument(
        '--' + START_TIME_ARG_NAME, type=str, required=True,
        help=TIME_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + END_TIME_ARG_NAME, type=str, required=True,
        help=TIME_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
        help=INPUT_DIR_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + DATA_SOURCE_ARG_NAME, type=str, required=True,
        help=DATA_SOURCE_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
        help=OUTPUT_DIR_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + TRACKING_SCALE_ARG_NAME, type=int, required=True,
        help=TRACKING_SCALE_HELP_STRING)

    return argument_parser_object


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _create_best_tracks(
        start_time_string=getattr(INPUT_ARG_OBJECT, START_TIME_ARG_NAME),
        end_time_string=getattr(INPUT_ARG_OBJECT, END_TIME_ARG_NAME),
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        data_source=getattr(INPUT_ARG_OBJECT, DATA_SOURCE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
        tracking_scale_metres2=getattr(
            INPUT_ARG_OBJECT, TRACKING_SCALE_ARG_NAME))
