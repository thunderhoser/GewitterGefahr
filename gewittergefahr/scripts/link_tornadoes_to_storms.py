"""Runs `linkage.link_tornadoes_to_storms`."""

import argparse
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import linkage

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SPC_DATE_ARG_NAME = 'spc_date_string'
TORNADO_DIR_ARG_NAME = 'input_tornado_dir_name'
TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
TRACKING_SCALE_ARG_NAME = 'tracking_scale_metres2'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Each storm cell on this day will be linked '
    'to zero or more wind observations.')

TORNADO_DIR_HELP_STRING = (
    'Name of directory with tornado observations.  Relevant files will be found'
    ' by `tornado_io.find_processed_file` and read by '
    '`tornado_io.read_processed_file`.')

TRACKING_DIR_HELP_STRING = (
    'Name of top-level tracking directory.  Files therein will be found by '
    '`storm_tracking_io.find_processed_files_one_spc_date` and read by '
    '`storm_tracking_io.read_processed_file`.')

TRACKING_SCALE_HELP_STRING = (
    'Tracking scale (minimum object area).  Used to find files in `{0:s}`.'
).format(TRACKING_DIR_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level directory for linkage files.  Files will be written by '
    '`linkage.write_linkage_file`, to locations therein determined by '
    '`linkage.find_linkage_file`.')

TOP_TORNADO_DIR_NAME_DEFAULT = (
    '/condo/swatwork/ralager/tornado_observations/processed')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TORNADO_DIR_ARG_NAME, type=str, required=False,
    default=TOP_TORNADO_DIR_NAME_DEFAULT, help=TORNADO_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_ARG_NAME, type=str, required=True,
    help=TRACKING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_SCALE_ARG_NAME, type=int, required=False,
    default=echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2,
    help=TRACKING_SCALE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(spc_date_string, tornado_dir_name, top_tracking_dir_name,
         tracking_scale_metres2, top_output_dir_name):
    """Runs `linkage.link_tornadoes_to_storms`.

    This is effectively the main method.

    :param spc_date_string: See documentation at top of file.
    :param tornado_dir_name: Same.
    :param top_tracking_dir_name: Same.
    :param tracking_scale_metres2: Same.
    :param top_output_dir_name: Same.
    """

    tracking_file_names, _ = tracking_io.find_processed_files_one_spc_date(
        spc_date_string=spc_date_string,
        data_source=tracking_utils.SEGMOTION_SOURCE_ID,
        top_processed_dir_name=top_tracking_dir_name,
        tracking_scale_metres2=tracking_scale_metres2,
        raise_error_if_missing=True)

    storm_to_tornadoes_table = linkage.link_storms_to_tornadoes(
        tracking_file_names=tracking_file_names,
        tornado_directory_name=tornado_dir_name)
    print SEPARATOR_STRING

    output_file_name = linkage.find_linkage_file(
        top_directory_name=top_output_dir_name,
        event_type_string=linkage.TORNADO_EVENT_STRING,
        spc_date_string=spc_date_string, raise_error_if_missing=False)

    print 'Writing linkages to: "{0:s}"...'.format(output_file_name)
    linkage.write_linkage_file(storm_to_events_table=storm_to_tornadoes_table,
                               pickle_file_name=output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        spc_date_string=getattr(INPUT_ARG_OBJECT, SPC_DATE_ARG_NAME),
        tornado_dir_name=getattr(INPUT_ARG_OBJECT, TORNADO_DIR_ARG_NAME),
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME),
        tracking_scale_metres2=getattr(
            INPUT_ARG_OBJECT, TRACKING_SCALE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
