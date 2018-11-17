"""Runs echo-top-based storm-tracking."""

import argparse
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import echo_top_tracking

RADAR_DIR_ARG_NAME = 'input_radar_dir_name'
ECHO_TOP_FIELD_ARG_NAME = 'echo_top_field_name'
ECHO_CLASSIFN_DIR_ARG_NAME = 'input_echo_classifn_dir_name'
MIN_ECHO_TOP_ARG_NAME = 'min_echo_top_km_asl'
MIN_GRID_CELLS_ARG_NAME = 'min_grid_cells_in_polygon'
OUTPUT_DIR_ARG_NAME = 'output_tracking_dir_name'
FIRST_SPC_DATE_ARG_NAME = 'first_spc_date_string'
LAST_SPC_DATE_ARG_NAME = 'last_spc_date_string'

RADAR_DIR_HELP_STRING = (
    'Name of top-level directory with radar files.  Files therein will be found'
    ' by `echo_top_tracking._find_input_radar_files` and read by '
    '`myrorss_and_mrms_io.read_data_from_sparse_grid_file`.')

ECHO_TOP_FIELD_HELP_STRING = (
    'Tracking will be based on this field.  Must be in the following list.'
    '\n{0:s}'
).format(str(radar_utils.ECHO_TOP_NAMES))

ECHO_CLASSIFN_DIR_HELP_STRING = (
    'Name of top-level directory with echo classifications.  If empty (""), '
    'echo classifications will not be used.  If non-empty, files therein will '
    'be found by `echo_classification.find_classification_file` and read by '
    '`echo_classification.read_classifications` and tracking will be run only '
    'on convective pixels.')

MIN_ECHO_TOP_HELP_STRING = (
    'Minimum echo top (km above sea level).  Only maxima with '
    '`{0:s}` >= `{1:s}` will be considered storm objects.  Smaller maxima will '
    'be thrown out.'
).format(ECHO_TOP_FIELD_ARG_NAME, MIN_ECHO_TOP_ARG_NAME)

MIN_GRID_CELLS_HELP_STRING = (
    'Minimum storm-object size.  Smaller objects will be thrown out.')

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Files will be written here by '
    '`echo_top_tracking._write_storm_objects`.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  This script will track storms in the period'
    ' `{0:s}`...`{1:s}`.'
).format(FIRST_SPC_DATE_ARG_NAME, LAST_SPC_DATE_ARG_NAME)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_DIR_ARG_NAME, type=str, required=True,
    help=RADAR_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ECHO_TOP_FIELD_ARG_NAME, type=str, required=False,
    default=radar_utils.ECHO_TOP_40DBZ_NAME, help=ECHO_TOP_FIELD_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ECHO_CLASSIFN_DIR_ARG_NAME, type=str, required=False, default='',
    help=ECHO_CLASSIFN_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_ECHO_TOP_ARG_NAME, type=int, required=False, default=4,
    help=MIN_ECHO_TOP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_GRID_CELLS_ARG_NAME, type=int, required=False, default=0,
    help=MIN_GRID_CELLS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)


def _run(top_radar_dir_name, echo_top_field_name, top_echo_classifn_dir_name,
         min_echo_top_km_asl, min_grid_cells_in_polygon, top_output_dir_name,
         first_spc_date_string, last_spc_date_string):
    """Runs echo-top-based storm-tracking.

    This is effectively the main method.

    :param top_radar_dir_name: See documentation at top of file.
    :param echo_top_field_name: Same.
    :param top_echo_classifn_dir_name: Same.
    :param min_echo_top_km_asl: Same.
    :param min_grid_cells_in_polygon: Same.
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
        echo_top_field_name=echo_top_field_name,
        top_echo_classifn_dir_name=top_echo_classifn_dir_name,
        min_echo_top_height_km_asl=min_echo_top_km_asl,
        min_grid_cells_in_polygon=min_grid_cells_in_polygon)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_radar_dir_name=getattr(INPUT_ARG_OBJECT, RADAR_DIR_ARG_NAME),
        echo_top_field_name=getattr(INPUT_ARG_OBJECT, ECHO_TOP_FIELD_ARG_NAME),
        top_echo_classifn_dir_name=getattr(
            INPUT_ARG_OBJECT, ECHO_CLASSIFN_DIR_ARG_NAME),
        min_echo_top_km_asl=float(
            getattr(INPUT_ARG_OBJECT, MIN_ECHO_TOP_ARG_NAME)),
        min_grid_cells_in_polygon=getattr(
            INPUT_ARG_OBJECT, MIN_GRID_CELLS_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_SPC_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_ARG_NAME)
    )
