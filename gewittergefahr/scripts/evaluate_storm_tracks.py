"""Evaluates a set of storm tracks."""

import argparse
import pandas
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import storm_tracking_eval as tracking_eval
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import time_conversion

MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

STORM_OBJECT_COLUMNS = [
    tracking_utils.PRIMARY_ID_COLUMN, tracking_utils.FULL_ID_COLUMN,
    tracking_utils.VALID_TIME_COLUMN, tracking_utils.SPC_DATE_COLUMN,
    tracking_utils.CENTROID_LATITUDE_COLUMN,
    tracking_utils.CENTROID_LONGITUDE_COLUMN,
    tracking_utils.ROWS_IN_STORM_COLUMN, tracking_utils.COLUMNS_IN_STORM_COLUMN
]

TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
FIRST_SPC_DATE_ARG_NAME = 'first_spc_date_string'
LAST_SPC_DATE_ARG_NAME = 'last_spc_date_string'
MYRORSS_DIR_ARG_NAME = 'input_myrorss_dir_name'
RADAR_FIELD_ARG_NAME = 'radar_field_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with tracks to evaluate.  Files therein will '
    'be found by `storm_tracking_io.find_processed_file` and read '
    '`storm_tracking_io.read_processed_file`.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Evaluation will be done for all SPC dates '
    'in the period `{0:s}`...`{1:s}`.'
).format(FIRST_SPC_DATE_ARG_NAME, LAST_SPC_DATE_ARG_NAME)

MYRORSS_DIR_HELP_STRING = (
    'Name of top-level directory with radar files.  Files therein will be found'
    ' by `myrorss_and_mrms_io.find_raw_file` and read by '
    '`myrorss_and_mrms_io.read_data_from_sparse_grid_file`.')

RADAR_FIELD_HELP_STRING = (
    'Name of field used to compute mismatch error.  Must be accepted by '
    '`radar_utils.check_field_name`.  Must not be a single-height reflectivity '
    'field.')

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by `storm_tracking_eval.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_ARG_NAME, type=str, required=True,
    help=TRACKING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MYRORSS_DIR_ARG_NAME, type=str, required=True,
    help=MYRORSS_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_FIELD_ARG_NAME, type=str, required=True,
    help=RADAR_FIELD_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(top_tracking_dir_name, first_spc_date_string, last_spc_date_string,
         top_myrorss_dir_name, radar_field_name, output_file_name):
    """Evaluates a set of storm tracks.

    This is effectively the main method.

    :param top_tracking_dir_name: See documentation at top of file.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param top_myrorss_dir_name: Same.
    :param radar_field_name: Same.
    :param output_file_name: Same.
    """

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    list_of_storm_object_tables = []
    print top_tracking_dir_name

    for this_spc_date_string in spc_date_strings:
        these_file_names = tracking_io.find_files_one_spc_date(
            top_tracking_dir_name=top_tracking_dir_name,
            tracking_scale_metres2=
            echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2,
            source_name=tracking_utils.SEGMOTION_NAME,
            spc_date_string=this_spc_date_string, raise_error_if_missing=False
        )[0]

        if len(these_file_names) == 0:
            continue

        this_storm_object_table = tracking_io.read_many_files(
            these_file_names
        )[STORM_OBJECT_COLUMNS]

        list_of_storm_object_tables.append(this_storm_object_table)

        if this_spc_date_string != spc_date_strings[-1]:
            print MINOR_SEPARATOR_STRING

        if len(list_of_storm_object_tables) == 1:
            continue

        list_of_storm_object_tables[-1] = list_of_storm_object_tables[-1].align(
            list_of_storm_object_tables[0], axis=1
        )[0]

    print SEPARATOR_STRING

    storm_object_table = pandas.concat(
        list_of_storm_object_tables, axis=0, ignore_index=True)

    evaluation_dict = tracking_eval.evaluate_tracks(
        storm_object_table=storm_object_table,
        top_myrorss_dir_name=top_myrorss_dir_name,
        radar_field_name=radar_field_name)

    print 'Writing results to: "{0:s}"...'.format(output_file_name)
    tracking_eval.write_file(evaluation_dict=evaluation_dict,
                             pickle_file_name=output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME),
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_SPC_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_ARG_NAME),
        top_myrorss_dir_name=getattr(INPUT_ARG_OBJECT, MYRORSS_DIR_ARG_NAME),
        radar_field_name=getattr(INPUT_ARG_OBJECT, RADAR_FIELD_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
