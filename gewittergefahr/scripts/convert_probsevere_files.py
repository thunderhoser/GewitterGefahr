"""Converts probSevere tracking files for one day.

This "conversion" is from the original ASCII format to GewitterGefahr format (a
pandas DataFrame stored in a Pickle file).
"""

import argparse
import numpy
from gewittergefahr.gg_io import probsevere_io
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import temporal_tracking
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

DATE_FORMAT = '%Y%m%d'
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M%S'
DUMMY_TRACKING_SCALE_METRES2 = int(numpy.round(numpy.pi * 1e8))

PROBSEVERE_DIR_ARG_NAME = 'input_probsevere_dir_name'
DATE_ARG_NAME = 'date_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PROBSEVERE_DIR_HELP_STRING = (
    'Name of top-level directory with raw probSevere files.')

DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  probSevere files will be converted for all time'
    ' steps on this date.')

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory (for converted probSevere files, to be '
    'written by `storm_tracking_io.write_processed_file`).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PROBSEVERE_DIR_ARG_NAME, type=str, required=True,
    help=PROBSEVERE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _convert_files(top_probsevere_dir_name, date_string, top_output_dir_name):
    """Converts probSevere tracking files for one day.

    :param top_probsevere_dir_name: See documentation at top of file.
    :param date_string: Same.
    :param top_output_dir_name: Same.
    """

    date_unix_sec = time_conversion.string_to_unix_sec(date_string, DATE_FORMAT)

    raw_file_names = probsevere_io.find_raw_files_one_day(
        top_directory_name=top_probsevere_dir_name, unix_time_sec=date_unix_sec,
        file_extension=probsevere_io.ASCII_FILE_EXTENSION,
        raise_error_if_all_missing=True)

    for this_raw_file_name in raw_file_names:
        print 'Reading data from "{0:s}"...'.format(this_raw_file_name)
        this_storm_object_table = probsevere_io.read_raw_file(
            this_raw_file_name)

        this_storm_object_table = (
            temporal_tracking.finish_segmotion_or_probsevere_ids(
                this_storm_object_table)
        )

        this_new_file_name = tracking_io.find_file(
            valid_time_unix_sec=probsevere_io.raw_file_name_to_time(
                this_raw_file_name),
            source_name=tracking_utils.PROBSEVERE_NAME,
            top_tracking_dir_name=top_output_dir_name,
            tracking_scale_metres2=DUMMY_TRACKING_SCALE_METRES2,
            raise_error_if_missing=False)

        print 'Writing data to "{0:s}"...'.format(this_new_file_name)
        tracking_io.write_file(
            storm_object_table=this_storm_object_table,
            pickle_file_name=this_new_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _convert_files(
        top_probsevere_dir_name=getattr(
            INPUT_ARG_OBJECT, PROBSEVERE_DIR_ARG_NAME),
        date_string=getattr(INPUT_ARG_OBJECT, DATE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
