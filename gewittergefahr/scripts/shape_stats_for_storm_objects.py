"""Computes shape statistics for each storm object."""

import argparse
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import shape_statistics as shape_stats
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SPC_DATE_INPUT_ARG = 'spc_date_string'
TRACKING_DIR_INPUT_ARG = 'input_tracking_dir_name'
TRACKING_SCALE_INPUT_ARG = 'tracking_scale_metres2'
OUTPUT_DIR_INPUT_ARG = 'output_dir_name'

SPC_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  Shape statistics'
    ' will be computed for all storm objects on this date.')

TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with storm-tracking data.')

TRACKING_SCALE_HELP_STRING = (
    'Tracking scale (minimum storm area).  Will be used to find input data.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  A single Pickle file, with shape statistics for'
    ' each storm object, will be written here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SPC_DATE_INPUT_ARG, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_SCALE_INPUT_ARG, type=int, required=False,
    default=echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2,
    help=TRACKING_SCALE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_INPUT_ARG, type=str, required=True,
    help=TRACKING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_INPUT_ARG, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _compute_shape_stats(
        spc_date_string, top_tracking_dir_name, tracking_scale_metres2,
        output_dir_name):
    """Computes shape statistics for each storm object.

    :param spc_date_string: SPC (Storm Prediction Center) date in format
        "yyyymmdd".  Shape statistics will be computed for all storm objects on
        this date.
    :param top_tracking_dir_name: Name of top-level directory with storm-
        tracking data.
    :param tracking_scale_metres2: Tracking scale (minimum storm area).  Will be
        used to find input data.
    :param output_dir_name: Name of output directory.  A single Pickle file,
        with shape statistics for each storm object, will be written here.
    """

    tracking_file_names = tracking_io.find_files_one_spc_date(
        spc_date_string=spc_date_string,
        source_name=tracking_utils.SEGMOTION_NAME,
        top_tracking_dir_name=top_tracking_dir_name,
        tracking_scale_metres2=tracking_scale_metres2
    )[0]

    storm_object_table = tracking_io.read_many_files(tracking_file_names)
    print SEPARATOR_STRING

    shape_statistic_table = shape_stats.get_stats_for_storm_objects(
        storm_object_table)
    print SEPARATOR_STRING

    shape_statistic_file_name = '{0:s}/shape_statistics_{1:s}.p'.format(
        output_dir_name, spc_date_string)

    print 'Writing shape statistics to: "{0:s}"...'.format(
        shape_statistic_file_name)
    shape_stats.write_stats_for_storm_objects(
        shape_statistic_table, shape_statistic_file_name)


def add_input_arguments(argument_parser_object):
    """Adds input args for this script to `argparse.ArgumentParser` object.

    :param argument_parser_object: `argparse.ArgumentParser` object, which may
        or may not already contain input args.
    :return: argument_parser_object: Same as input object, but with new input
        args added.
    """

    argument_parser_object.add_argument(
        '--' + TRACKING_SCALE_INPUT_ARG, type=int, required=False,
        default=echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2,
        help=TRACKING_SCALE_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + TRACKING_DIR_INPUT_ARG, type=str, required=True,
        help=TRACKING_DIR_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + OUTPUT_DIR_INPUT_ARG, type=str, required=True,
        help=OUTPUT_DIR_HELP_STRING)

    return argument_parser_object


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _compute_shape_stats(
        spc_date_string=getattr(INPUT_ARG_OBJECT, SPC_DATE_INPUT_ARG),
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_INPUT_ARG),
        tracking_scale_metres2=getattr(
            INPUT_ARG_OBJECT, TRACKING_SCALE_INPUT_ARG),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_INPUT_ARG)
    )
