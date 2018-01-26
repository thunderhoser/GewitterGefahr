"""Runs GridRad storm-tracking algorithm with default parameters.

This algorithm is discussed in Section 3c of Homeyer et al. (2017) and the
documentation for `gridrad_tracking.py`.

--- REFERENCES ---

Homeyer, C.R., and J.D. McAuliffe, and K.M. Bedka, 2017: "On the development of
    above-anvil cirrus plumes in extratropical convection". Journal of the
    Atmospheric Sciences, 74 (5), 1617-1633.
"""

import argparse
from gewittergefahr.gg_utils import gridrad_tracking

# TODO(thunderhoser): All input args to GridRad tracking should be input args to
# this script.  In other words, allow for other than default params.

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FIRST_SPC_DATE_INPUT_ARG = 'first_spc_date_string'
LAST_SPC_DATE_INPUT_ARG = 'last_spc_date_string'
RADAR_DIR_INPUT_ARG = 'input_radar_dir_name'
TRACKING_DIR_INPUT_ARG = 'output_tracking_dir_name'

SPC_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  Tracking will be'
    ' run for all dates from `{0:s}`...`{1:s}`.').format(
        FIRST_SPC_DATE_INPUT_ARG, LAST_SPC_DATE_INPUT_ARG)
RADAR_DIR_HELP_STRING = 'Name of top-level directory with radar data.'
TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory for storm-tracking data.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_INPUT_ARG, type=str, required=True,
    help=SPC_DATE_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_INPUT_ARG, type=str, required=True,
    help=SPC_DATE_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_DIR_INPUT_ARG, type=str, required=True,
    help=RADAR_DIR_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_INPUT_ARG, type=str, required=True,
    help=TRACKING_DIR_HELP_STRING)


def _run_gridrad_tracking(
        first_spc_date_string, last_spc_date_string, top_radar_dir_name,
        top_tracking_dir_name):
    """Runs GridRad storm-tracking algorithm with default parameters.

    :param first_spc_date_string: SPC (Storm Prediction Center) date in format
        "yyyymmdd".  Tracking will be run for all dates from
        `first_spc_date_string`...`last_spc_date_string`.
    :param last_spc_date_string: See above.
    :param top_radar_dir_name: [input] Name of top-level directory with radar
        data.
    :param top_tracking_dir_name: [output] Name of top-level directory for
        storm-tracking data.
    """

    storm_object_table, file_dictionary = gridrad_tracking.run_tracking(
        top_radar_dir_name=top_radar_dir_name,
        top_tracking_dir_name=top_tracking_dir_name,
        start_spc_date_string=first_spc_date_string,
        end_spc_date_string=last_spc_date_string)
    print SEPARATOR_STRING

    gridrad_tracking.write_storm_objects(storm_object_table, file_dictionary)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    FIRST_SPC_DATE_STRING = getattr(
        INPUT_ARG_OBJECT, FIRST_SPC_DATE_INPUT_ARG)
    LAST_SPC_DATE_STRING = getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_INPUT_ARG)
    TOP_RADAR_DIR_NAME = getattr(INPUT_ARG_OBJECT, RADAR_DIR_INPUT_ARG)
    TOP_TRACKING_DIR_NAME = getattr(INPUT_ARG_OBJECT, TRACKING_DIR_INPUT_ARG)

    _run_gridrad_tracking(
        first_spc_date_string=FIRST_SPC_DATE_STRING,
        last_spc_date_string=LAST_SPC_DATE_STRING,
        top_radar_dir_name=TOP_RADAR_DIR_NAME,
        top_tracking_dir_name=TOP_TRACKING_DIR_NAME)
