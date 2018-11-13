"""Runs echo-top-based storm-tracking."""

import argparse
from gewittergefahr.gg_utils import echo_top_tracking

RADAR_DIR_ARG_NAME = 'input_radar_dir_name'
ECHO_CLASSIFN_DIR_ARG_NAME = 'input_echo_classifn_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_tracking_dir_name'
FIRST_SPC_DATE_ARG_NAME = 'first_spc_date_string'
LAST_SPC_DATE_ARG_NAME = 'last_spc_date_string'

RADAR_DIR_HELP_STRING = (
    'Name of top-level directory with radar files.  Files therein will be found'
    ' by `myrorss_and_mrms_io.find_raw_files_one_spc_date`.')

ECHO_CLASSIFN_DIR_HELP_STRING = (
    'Name of top-level directory with echo classifications.  If empty (""), '
    'echo classifications will not be used.  If non-empty, files therein will '
    'be found by `echo_classification.find_classification_file` and read by '
    '`echo_classification.read_classifications` and tracking will be run only '
    'on convective pixels.')

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Files will be written by '
    '`storm_tracking_io.write_processed_file`, to locations therein determined '
    'by `storm_tracking_io.find_processed_file`.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Echo-top-tracking will be done for dates '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_SPC_DATE_ARG_NAME, LAST_SPC_DATE_ARG_NAME)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_DIR_ARG_NAME, type=str, required=True,
    help=RADAR_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ECHO_CLASSIFN_DIR_ARG_NAME, type=str, required=False, default='',
    help=ECHO_CLASSIFN_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)


def _run(top_radar_dir_name, top_echo_classifn_dir_name, top_output_dir_name,
         first_spc_date_string, last_spc_date_string):
    """Runs echo-top-based storm-tracking.

    This is effectively the main method.

    :param top_radar_dir_name: See documentation at top of file.
    :param top_echo_classifn_dir_name: Same.
    :param top_output_dir_name: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    """

    if top_echo_classifn_dir_name in ['', 'None']:
        top_echo_classifn_dir_name = None

    echo_top_tracking.run_tracking(
        top_radar_dir_name=top_radar_dir_name,
        top_output_dir_name=top_output_dir_name,
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string,
        top_echo_classifn_dir_name=top_echo_classifn_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_radar_dir_name=getattr(INPUT_ARG_OBJECT, RADAR_DIR_ARG_NAME),
        top_echo_classifn_dir_name=getattr(
            INPUT_ARG_OBJECT, ECHO_CLASSIFN_DIR_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_SPC_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_ARG_NAME)
    )
