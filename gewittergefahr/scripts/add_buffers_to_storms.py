"""Creates one or more distance buffers around each storm object (polygon)."""

import argparse
import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_tracking_dir_name'
TRACKING_SCALE_ARG_NAME = 'tracking_scale_metres2'
FIRST_DATE_ARG_NAME = 'first_spc_date_string'
LAST_DATE_ARG_NAME = 'last_spc_date_string'
MIN_DISTANCES_ARG_NAME = 'min_distances_metres'
MAX_DISTANCES_ARG_NAME = 'max_distances_metres'
OUTPUT_DIR_ARG_NAME = 'output_tracking_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory with input tracking data.  Files therein will '
    'be found by `storm_tracking_io.find_processed_files_one_spc_date` and read'
    ' by `storm_tracking_io.read_processed_file`.')

TRACKING_SCALE_HELP_STRING = (
    'Tracking scale (used only to find files in `{0:s}`).'
).format(INPUT_DIR_ARG_NAME)

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  This script will be run on storm objects in'
    ' the period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

MIN_DISTANCES_HELP_STRING = (
    'List of minimum distances (one for each buffer).  Negative distance means '
    'that the buffer includes the storm.')

MAX_DISTANCES_HELP_STRING = 'List of max distances (one for each buffer).'

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Tracking data with buffers will be '
    'written here by `storm_tracking_io.write_processed_file`, to exact '
    'locations determined by `storm_tracking_io.find_processed_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_SCALE_ARG_NAME, type=int, required=False,
    default=echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2,
    help=TRACKING_SCALE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_DISTANCES_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=MIN_DISTANCES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_DISTANCES_ARG_NAME, type=int, nargs='+', required=False,
    default=[10000], help=MAX_DISTANCES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(top_input_dir_name, tracking_scale_metres2, first_spc_date_string,
         last_spc_date_string, min_distances_metres, max_distances_metres,
         top_output_dir_name):
    """Creates one or more distance buffers around each storm object (polygon).

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param tracking_scale_metres2: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param min_distances_metres: Same.
    :param max_distances_metres: Same.
    :param top_output_dir_name: Same.
    """

    min_distances_metres[min_distances_metres < 0] = numpy.nan

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    for this_spc_date_string in spc_date_strings:
        these_input_file_names = tracking_io.find_files_one_spc_date(
            top_tracking_dir_name=top_input_dir_name,
            tracking_scale_metres2=tracking_scale_metres2,
            source_name=tracking_utils.SEGMOTION_NAME,
            spc_date_string=this_spc_date_string,
            raise_error_if_missing=False
        )[0]

        if len(these_input_file_names) == 0:
            continue

        for this_input_file_name in these_input_file_names:
            print('Reading input tracks from: "{0:s}"...'.format(
                this_input_file_name))

            this_storm_object_table = tracking_io.read_file(
                this_input_file_name)

            this_storm_object_table = (
                tracking_utils.create_distance_buffers(
                    storm_object_table=this_storm_object_table,
                    min_distances_metres=min_distances_metres,
                    max_distances_metres=max_distances_metres)
            )

            this_output_file_name = tracking_io.find_file(
                top_tracking_dir_name=top_output_dir_name,
                tracking_scale_metres2=tracking_scale_metres2,
                source_name=tracking_utils.SEGMOTION_NAME,
                valid_time_unix_sec=tracking_io.file_name_to_time(
                    this_input_file_name),
                spc_date_string=this_spc_date_string,
                raise_error_if_missing=False)

            print('Writing input tracks + buffers to: "{0:s}"...\n'.format(
                this_output_file_name))

            tracking_io.write_file(
                storm_object_table=this_storm_object_table,
                pickle_file_name=this_output_file_name)

        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        tracking_scale_metres2=getattr(
            INPUT_ARG_OBJECT, TRACKING_SCALE_ARG_NAME),
        first_spc_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        min_distances_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, MIN_DISTANCES_ARG_NAME), dtype=float
        ),
        max_distances_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_DISTANCES_ARG_NAME), dtype=float
        ),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
