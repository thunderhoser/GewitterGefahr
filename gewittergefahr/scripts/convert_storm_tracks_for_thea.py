"""Converts storm tracks to a single CSV file for Thea."""

import argparse
import pandas
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import best_tracks
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

FIRST_SPC_DATE_ARG_NAME = 'first_spc_date_string'
LAST_SPC_DATE_ARG_NAME = 'last_spc_date_string'
TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
TRACKING_SCALE_ARG_NAME = 'tracking_scale_metres2'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

SPC_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  Will convert '
    'tracking dates for all dates in `{0:s}`...`{1:s}`.'
).format(FIRST_SPC_DATE_ARG_NAME, LAST_SPC_DATE_ARG_NAME)
TRACKING_DIR_HELP_STRING = 'Name of top-level directory with storm tracks.'
TRACKING_SCALE_HELP_STRING = (
    'Tracking scale (minimum storm area).  This will be used only to find input'
    ' files.')
OUTPUT_FILE_HELP_STRING = 'Path to output file (CSV for Thea).'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_ARG_NAME, type=str, required=True,
    help=TRACKING_DIR_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_SCALE_ARG_NAME, type=int, required=False,
    default=echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2,
    help=TRACKING_SCALE_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _write_csv_file_for_thea(
        first_spc_date_string, last_spc_date_string, top_tracking_dir_name,
        tracking_scale_metres2, output_file_name):
    """Converts storm tracks to a single CSV file for Thea.

    :param first_spc_date_string: See documentation at top of file.
    :param last_spc_date_string: Same.
    :param top_tracking_dir_name: Same.
    :param tracking_scale_metres2: Same.
    :param output_file_name: Same.
    """

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    tracking_file_names = []
    for this_spc_date_string in spc_date_strings:
        (these_tracking_file_names, _
        ) = tracking_io.find_processed_files_one_spc_date(
            spc_date_string=this_spc_date_string,
            data_source=tracking_utils.SEGMOTION_SOURCE_ID,
            top_processed_dir_name=top_tracking_dir_name,
            tracking_scale_metres2=tracking_scale_metres2)
        tracking_file_names += these_tracking_file_names

    num_files = len(tracking_file_names)
    list_of_storm_object_tables = [None] * num_files

    for i in range(num_files):
        print 'Reading {0:d}th of {1:d} input files: "{2:s}"...'.format(
            i + 1, num_files, tracking_file_names[i])

        list_of_storm_object_tables[i] = tracking_io.read_processed_file(
            tracking_file_names[i])
        if i == 0:
            continue

        list_of_storm_object_tables[i], _ = (
            list_of_storm_object_tables[i].align(
                list_of_storm_object_tables[0], axis=1))

    storm_object_table = pandas.concat(
        list_of_storm_object_tables, axis=0, ignore_index=True)

    print 'Writing all storm tracks to "{0:s}"...'.format(output_file_name)
    best_tracks.write_simple_output_for_thea(
        storm_object_table, output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    FIRST_SPC_DATE_STRING = getattr(
        INPUT_ARG_OBJECT, FIRST_SPC_DATE_ARG_NAME)
    LAST_SPC_DATE_STRING = getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_ARG_NAME)
    TOP_TRACKING_DIR_NAME = getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME)
    TRACKING_SCALE_METRES2 = getattr(INPUT_ARG_OBJECT, TRACKING_SCALE_ARG_NAME)
    OUTPUT_FILE_NAME = getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)

    _write_csv_file_for_thea(
        first_spc_date_string=FIRST_SPC_DATE_STRING,
        last_spc_date_string=LAST_SPC_DATE_STRING,
        top_tracking_dir_name=TOP_TRACKING_DIR_NAME,
        tracking_scale_metres2=TRACKING_SCALE_METRES2,
        output_file_name=OUTPUT_FILE_NAME)
