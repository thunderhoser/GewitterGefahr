"""Finds which storms are linked to an NWS tornado warning."""

import pickle
import argparse
import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import temporal_tracking
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.scripts import link_warnings_to_storms as link_warnings

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_SECONDS_PER_DAY = 86400
LINKED_SECONDARY_IDS_KEY = link_warnings.LINKED_SECONDARY_IDS_KEY

STORM_METAFILE_ARG_NAME = 'input_storm_metafile_name'
WARNING_DIR_ARG_NAME = 'input_warning_dir_name'

STORM_METAFILE_HELP_STRING = (
    'Path to file with storm IDs and times (will be read by '
    '`storm_tracking_io.read_ids_and_times`).  Will search for warnings linked'
    'only to these storms.'
)
WARNING_DIR_HELP_STRING = (
    'Name of directory with warning files (created by '
    'link_warnings_to_storms.py).  File for SPC date yyyymmdd should be named '
    '"tornado_warnings_[yyyymmdd].p".'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + STORM_METAFILE_ARG_NAME, type=str, required=True,
    help=STORM_METAFILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + WARNING_DIR_ARG_NAME, type=str, required=True,
    help=WARNING_DIR_HELP_STRING
)


def _run(storm_metafile_name, warning_dir_name):
    """Finds which storms are linked to an NWS tornado warning.

    This is effectively the main method.

    :param storm_metafile_name: See documentation at top of file.
    :param warning_dir_name: Same.
    """

    print('Reading storm metadata from: "{0:s}"...'.format(storm_metafile_name))
    full_storm_id_strings, valid_times_unix_sec = (
        tracking_io.read_ids_and_times(storm_metafile_name)
    )
    secondary_id_strings = (
        temporal_tracking.full_to_partial_ids(full_storm_id_strings)[-1]
    )

    these_times_unix_sec = numpy.concatenate((
        valid_times_unix_sec,
        valid_times_unix_sec - NUM_SECONDS_PER_DAY,
        valid_times_unix_sec + NUM_SECONDS_PER_DAY
    ))

    spc_date_strings = [
        time_conversion.time_to_spc_date_string(t) for t in these_times_unix_sec
    ]
    spc_date_strings = numpy.unique(numpy.array(spc_date_strings))

    linked_secondary_id_strings = []

    for this_spc_date_string in spc_date_strings:
        this_file_name = '{0:s}/tornado_warnings_{1:s}.p'.format(
            warning_dir_name, this_spc_date_string
        )
        print('Reading warnings from: "{0:s}"...'.format(this_file_name))

        this_file_handle = open(this_file_name, 'rb')
        this_warning_table = pickle.load(this_file_handle)
        this_file_handle.close()

        this_num_warnings = len(this_warning_table.index)

        for k in range(this_num_warnings):
            linked_secondary_id_strings += (
                this_warning_table[LINKED_SECONDARY_IDS_KEY].values[k]
            )

    print(SEPARATOR_STRING)

    storm_warned_flags = numpy.array([
        s in linked_secondary_id_strings for s in secondary_id_strings
    ], dtype=bool)

    print((
        '{0:d} of {1:d} storm objects are linked to an NWS tornado warning!'
    ).format(
        numpy.sum(storm_warned_flags), len(storm_warned_flags)
    ))


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        storm_metafile_name=getattr(INPUT_ARG_OBJECT, STORM_METAFILE_ARG_NAME),
        warning_dir_name=getattr(INPUT_ARG_OBJECT, WARNING_DIR_ARG_NAME)
    )
