"""Uses GridRad data to compute radar statistics for each storm object."""

import argparse
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import radar_statistics
from gewittergefahr.gg_utils import file_system_utils

# TODO(thunderhoser): All input args to
# `radar_statistics.compute_gridrad_stats_for_storm_objects` should be input
# args to this script.

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SPC_DATE_INPUT_ARG = 'spc_date_string'
TRACKING_DIR_INPUT_ARG = 'input_tracking_dir_name'
TRACKING_SCALE_INPUT_ARG = 'tracking_scale_metres2'
GRIDRAD_DIR_INPUT_ARG = 'input_gridrad_dir_name'
OUTPUT_DIR_INPUT_ARG = 'output_dir_name'

SPC_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  Radar stats will'
    ' be computed for all storm objects on this date.')
TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with storm-tracking files.  Storm objects will'
    ' be read from here.')
TRACKING_SCALE_HELP_STRING = (
    'Tracking scale (minimum storm area).  Will be used to find tracking '
    'files.')
GRIDRAD_DIR_HELP_STRING = 'Name of top-level directory with GridRad files.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  A single Pickle file, with radar stats for each'
    ' storm object, will be written here.')

DEFAULT_TRACKING_DIR_NAME = '/condo/swatcommon/common/gridrad/echo_top_tracking'
DEFAULT_GRIDRAD_DIR_NAME = '/condo/swatcommon/common/gridrad/native_format'
DEFAULT_OUTPUT_DIR_NAME = '/condo/swatcommon/common/gridrad/radar_statistics'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SPC_DATE_INPUT_ARG, type=str, required=True,
    help=SPC_DATE_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_INPUT_ARG, type=str, required=False,
    default=DEFAULT_TRACKING_DIR_NAME, help=TRACKING_DIR_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_SCALE_INPUT_ARG, type=int, required=False,
    default=echo_top_tracking.DEFAULT_STORM_OBJECT_AREA_METRES2,
    help=TRACKING_SCALE_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + GRIDRAD_DIR_INPUT_ARG, type=str, required=False,
    default=DEFAULT_GRIDRAD_DIR_NAME, help=GRIDRAD_DIR_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_INPUT_ARG, type=str, required=False,
    default=DEFAULT_OUTPUT_DIR_NAME, help=OUTPUT_DIR_HELP_STRING)


def _compute_radar_stats_from_gridrad(
        spc_date_string, top_tracking_dir_name, tracking_scale_metres2,
        top_gridrad_dir_name, output_dir_name):
    """Uses GridRad data to compute radar statistics for each storm object.

    :param spc_date_string: SPC (Storm Prediction Center) date in format
        "yyyymmdd".  Radar stats will be computed for all storm objects on this
        date.
    :param top_tracking_dir_name: Name of top-level directory with storm-
        tracking files.  Storm objects will be read from here.
    :param tracking_scale_metres2: Tracking scale (minimum storm area).  Will be
        used to find tracking files.
    :param top_gridrad_dir_name: Name of top-level directory with GridRad files.
    :param output_dir_name: Name of output directory.  A single Pickle file,
        with radar stats for each storm object, will be written here.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    tracking_file_names = tracking_io.find_processed_files_one_spc_date(
        spc_date_string=spc_date_string,
        data_source=tracking_utils.SEGMOTION_SOURCE_ID,
        top_processed_dir_name=top_tracking_dir_name,
        tracking_scale_metres2=tracking_scale_metres2)

    storm_object_table = tracking_io.read_many_processed_files(
        tracking_file_names)
    print SEPARATOR_STRING

    storm_radar_statistic_table = (
        radar_statistics.compute_gridrad_stats_for_storm_objects(
            storm_object_table=storm_object_table,
            top_radar_directory_name=top_gridrad_dir_name))
    print SEPARATOR_STRING

    output_file_name = '{0:s}/radar_stats_for_storm_objects_{1:s}.p'.format(
        output_dir_name, spc_date_string)
    print 'Writing radar statistics to file: "{0:s}"...'.format(
        output_file_name)
    radar_statistics.write_stats_for_storm_objects(
        storm_radar_statistic_table, output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    SPC_DATE_STRING = getattr(INPUT_ARG_OBJECT, SPC_DATE_INPUT_ARG)
    TOP_TRACKING_DIR_NAME = getattr(INPUT_ARG_OBJECT, TRACKING_DIR_INPUT_ARG)
    TRACKING_SCALE_METRES2 = getattr(INPUT_ARG_OBJECT, TRACKING_SCALE_INPUT_ARG)
    TOP_GRIDRAD_DIR_NAME = getattr(INPUT_ARG_OBJECT, GRIDRAD_DIR_INPUT_ARG)
    OUTPUT_DIR_NAME = getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_INPUT_ARG)

    _compute_radar_stats_from_gridrad(
        spc_date_string=SPC_DATE_STRING,
        top_tracking_dir_name=TOP_TRACKING_DIR_NAME,
        tracking_scale_metres2=TRACKING_SCALE_METRES2,
        top_gridrad_dir_name=TOP_GRIDRAD_DIR_NAME,
        output_dir_name=OUTPUT_DIR_NAME)
