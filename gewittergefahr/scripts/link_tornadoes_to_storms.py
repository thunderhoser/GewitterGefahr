"""Runs `linkage.link_tornadoes_to_storms`."""

import argparse
import numpy
from gewittergefahr.gg_io import tornado_io
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import linkage

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TORNADO_DIR_ARG_NAME = 'input_tornado_dir_name'
TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
TRACKING_SCALE_ARG_NAME = 'tracking_scale_metres2'
GENESIS_ONLY_ARG_NAME = 'genesis_only'
FIRST_SPC_DATE_ARG_NAME = 'first_spc_date_string'
LAST_SPC_DATE_ARG_NAME = 'last_spc_date_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

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

GENESIS_ONLY_HELP_STRING = (
    'Boolean flag.  If 1, will link only tornadogenesis events to storms.  If '
    '0, will link all points of tornado occurrence to storms.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  This script will operate independently on'
    ' each set of consecutive days in `{0:s}`...`{1:s}`.'
).format(FIRST_SPC_DATE_ARG_NAME, LAST_SPC_DATE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level directory for linkage files.  Files will be written by '
    '`linkage.write_linkage_file`, to locations therein determined by '
    '`linkage.find_linkage_file`.')

TOP_TORNADO_DIR_NAME_DEFAULT = (
    '/condo/swatwork/ralager/tornado_observations/processed')

INPUT_ARG_PARSER = argparse.ArgumentParser()
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
    '--' + GENESIS_ONLY_ARG_NAME, type=int, required=False, default=1,
    help=GENESIS_ONLY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _link_tornadoes_one_period(
        tracking_file_names, tornado_dir_name, genesis_only,
        top_output_dir_name):
    """Links tornadoes to storms for one continuous period.

    :param tracking_file_names: 1-D list of paths to tracking files.  Each will
        be read by `storm_tracking_io.read_processed_file`.
    :param tornado_dir_name: See documentation at top of file.
    :param genesis_only: Same.
    :param top_output_dir_name: Same.
    """

    storm_to_tornadoes_table, tornado_to_storm_table, metadata_dict = (
        linkage.link_storms_to_tornadoes(
            tracking_file_names=tracking_file_names,
            tornado_directory_name=tornado_dir_name, genesis_only=genesis_only)
    )
    print(SEPARATOR_STRING)

    event_type_string = (
        linkage.TORNADOGENESIS_EVENT_STRING if genesis_only
        else linkage.TORNADO_EVENT_STRING
    )

    spc_date_string_by_storm_object = [
        time_conversion.time_to_spc_date_string(t) for t in
        storm_to_tornadoes_table[tracking_utils.VALID_TIME_COLUMN].values
    ]

    unique_spc_date_strings, orig_to_unique_indices = numpy.unique(
        numpy.array(spc_date_string_by_storm_object), return_inverse=True
    )

    for i in range(len(unique_spc_date_strings)):
        this_output_file_name = linkage.find_linkage_file(
            top_directory_name=top_output_dir_name,
            event_type_string=event_type_string,
            spc_date_string=unique_spc_date_strings[i],
            raise_error_if_missing=False)

        print('Writing linkages to: "{0:s}"...'.format(this_output_file_name))

        these_storm_object_rows = numpy.where(orig_to_unique_indices == i)[0]
        these_storm_times_unix_sec = storm_to_tornadoes_table[
            tracking_utils.VALID_TIME_COLUMN].values[these_storm_object_rows]

        this_min_time_unix_sec = (
            numpy.min(these_storm_times_unix_sec) -
            metadata_dict[linkage.MAX_TIME_BEFORE_START_KEY]
        )
        this_max_time_unix_sec = (
            numpy.max(these_storm_times_unix_sec) +
            metadata_dict[linkage.MAX_TIME_AFTER_END_KEY]
        )

        if genesis_only:
            these_event_rows = numpy.where(numpy.logical_and(
                tornado_to_storm_table[linkage.EVENT_TIME_COLUMN].values >=
                this_min_time_unix_sec,
                tornado_to_storm_table[linkage.EVENT_TIME_COLUMN].values <=
                this_max_time_unix_sec
            ))[0]

            this_tornado_to_storm_table = tornado_to_storm_table.iloc[
                these_event_rows]
        else:
            column_dict_old_to_new = {
                linkage.EVENT_TIME_COLUMN: tornado_io.TIME_COLUMN,
                linkage.EVENT_LATITUDE_COLUMN: tornado_io.LATITUDE_COLUMN,
                linkage.EVENT_LONGITUDE_COLUMN: tornado_io.LONGITUDE_COLUMN
            }

            this_tornado_table = tornado_to_storm_table.rename(
                columns=column_dict_old_to_new, inplace=False)

            this_tornado_table = tornado_io.segments_to_tornadoes(
                this_tornado_table)

            this_tornado_table = tornado_io.subset_tornadoes(
                tornado_table=this_tornado_table,
                min_time_unix_sec=this_min_time_unix_sec,
                max_time_unix_sec=this_max_time_unix_sec)

            these_tornado_id_strings = this_tornado_table[
                tornado_io.TORNADO_ID_COLUMN].values

            this_tornado_to_storm_table = tornado_to_storm_table.loc[
                tornado_to_storm_table[tornado_io.TORNADO_ID_COLUMN].isin(
                    these_tornado_id_strings)
            ]

        linkage.write_linkage_file(
            pickle_file_name=this_output_file_name,
            storm_to_events_table=storm_to_tornadoes_table.iloc[
                these_storm_object_rows],
            metadata_dict=metadata_dict,
            tornado_to_storm_table=this_tornado_to_storm_table)
        print(SEPARATOR_STRING)


def _run(tornado_dir_name, top_tracking_dir_name, tracking_scale_metres2,
         genesis_only, first_spc_date_string, last_spc_date_string,
         top_output_dir_name):
    """Runs `linkage.link_tornadoes_to_storms`.

    This is effectively the main method.

    :param tornado_dir_name: See documentation at top of file.
    :param top_tracking_dir_name: Same.
    :param tracking_scale_metres2: Same.
    :param genesis_only: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param top_output_dir_name: Same.
    """

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    tracking_file_names = []

    for this_spc_date_string in spc_date_strings:
        these_file_names = tracking_io.find_files_one_spc_date(
            top_tracking_dir_name=top_tracking_dir_name,
            tracking_scale_metres2=tracking_scale_metres2,
            source_name=tracking_utils.SEGMOTION_NAME,
            spc_date_string=this_spc_date_string, raise_error_if_missing=False
        )[0]

        if len(these_file_names) == 0:
            if len(tracking_file_names) > 0:
                _link_tornadoes_one_period(
                    tracking_file_names=tracking_file_names,
                    tornado_dir_name=tornado_dir_name,
                    genesis_only=genesis_only,
                    top_output_dir_name=top_output_dir_name)

                print(SEPARATOR_STRING)
                tracking_file_names = []

            continue

        tracking_file_names += these_file_names

    _link_tornadoes_one_period(
        tracking_file_names=tracking_file_names,
        tornado_dir_name=tornado_dir_name,
        genesis_only=genesis_only,
        top_output_dir_name=top_output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        tornado_dir_name=getattr(INPUT_ARG_OBJECT, TORNADO_DIR_ARG_NAME),
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME),
        tracking_scale_metres2=getattr(
            INPUT_ARG_OBJECT, TRACKING_SCALE_ARG_NAME),
        genesis_only=bool(getattr(INPUT_ARG_OBJECT, GENESIS_ONLY_ARG_NAME)),
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_SPC_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
