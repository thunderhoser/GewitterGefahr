"""Removes short-lived storms from dataset."""

import argparse
import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import echo_top_tracking

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_tracking_dir_name'
FIRST_SPC_DATE_ARG_NAME = 'first_spc_date_string'
LAST_SPC_DATE_ARG_NAME = 'last_spc_date_string'
MIN_DURATION_ARG_NAME = 'min_duration_seconds'
OUTPUT_DIR_ARG_NAME = 'output_tracking_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory with original tracks (before removing short '
    'ones).  Files therein will be found by '
    '`storm_tracking_io.find_processed_file` and read by '
    '`storm_tracking_io.read_processed_file`.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  This script will operate on the period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_SPC_DATE_ARG_NAME, LAST_SPC_DATE_ARG_NAME)

MIN_DURATION_HELP_STRING = (
    'Minimum storm duration.  Shorter-lived storms will be thrown out.')

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level directory for new tracks (after removing short ones).  '
    'Files will be written by `storm_tracking_io.write_processed_file`, to '
    'locations therein determined by `storm_tracking_io.find_processed_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_DURATION_ARG_NAME, type=int, required=True,
    help=MIN_DURATION_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _remove_storms_one_period(input_file_names, min_duration_seconds,
                              top_output_dir_name):
    """Removes short-lived storms for one continuous period.

    :param input_file_names: 1-D list of paths to input files (will be read by
        `storm_tracking_io.read_processed_file`).
    :param min_duration_seconds: See documentation at top of file.
    :param top_output_dir_name: Same.
    """

    if len(input_file_names) == 0:
        return

    storm_object_table = tracking_io.read_many_processed_files(input_file_names)
    print MINOR_SEPARATOR_STRING

    orig_num_storm_objects = len(storm_object_table.index)
    storm_object_table = echo_top_tracking.remove_short_lived_tracks(
        storm_object_table=storm_object_table,
        min_duration_seconds=min_duration_seconds)
    num_storm_objects = len(storm_object_table.index)

    print (
        'Number of original storm objects = {0:d} ... num with duration '
        '> {1:d} seconds = {2:d}\n'
    ).format(orig_num_storm_objects, min_duration_seconds, num_storm_objects)

    valid_times_unix_sec = numpy.sort(numpy.array(
        [tracking_io.processed_file_name_to_time(f) for f in input_file_names],
        dtype=int
    ))

    for this_time_unix_sec in valid_times_unix_sec:
        this_output_file_name = tracking_io.find_processed_file(
            top_processed_dir_name=top_output_dir_name,
            unix_time_sec=this_time_unix_sec,
            spc_date_string=time_conversion.time_to_spc_date_string(
                this_time_unix_sec),
            tracking_scale_metres2=
            echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2,
            data_source=tracking_utils.SEGMOTION_SOURCE_ID,
            raise_error_if_missing=False)

        print 'Writing new data to: "{0:s}"...'.format(this_output_file_name)
        tracking_io.write_processed_file(
            storm_object_table=storm_object_table.loc[
                storm_object_table[tracking_utils.TIME_COLUMN] ==
                this_time_unix_sec
            ],
            pickle_file_name=this_output_file_name
        )


def _run(top_input_dir_name, first_spc_date_string, last_spc_date_string,
         min_duration_seconds, top_output_dir_name):
    """Removes short-lived storms from dataset.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param min_duration_seconds: Same.
    :param top_output_dir_name: Same.
    """

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    input_file_names = []

    for this_spc_date_string in spc_date_strings:
        these_file_names = tracking_io.find_processed_files_one_spc_date(
            top_processed_dir_name=top_input_dir_name,
            tracking_scale_metres2=
            echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2,
            data_source=tracking_utils.SEGMOTION_SOURCE_ID,
            spc_date_string=this_spc_date_string, raise_error_if_missing=False
        )[0]

        if len(these_file_names) == 0:
            _remove_storms_one_period(
                input_file_names=input_file_names,
                min_duration_seconds=min_duration_seconds,
                top_output_dir_name=top_output_dir_name)

            print SEPARATOR_STRING

            input_file_names = []
            continue

        input_file_names += these_file_names

    _remove_storms_one_period(
        input_file_names=input_file_names,
        min_duration_seconds=min_duration_seconds,
        top_output_dir_name=top_output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_SPC_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_ARG_NAME),
        min_duration_seconds=getattr(INPUT_ARG_OBJECT, MIN_DURATION_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
