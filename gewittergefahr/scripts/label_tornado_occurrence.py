"""Creates tornado label per storm object, lead-time window, & distance window.

Specifically, this script runs `labels.label_tornado_occurrence` with different
ranges of lead time and linkage distance.

--- EXAMPLES ---

Suppose that you want three lead-time windows (0-15, 15-30, and 30-60 minutes)
and two distance windows (0-10 km and > 10 km).  You could call this script as
follows (100 000 metres is a very large linkage distance, so basically functions
as infinity).

label_tornado_occurrence.py
--storm_to_tornadoes_file_name="${STORM_TO_TORNADOES_FILE_NAME}"
--min_lead_times_sec 0 900 1800 --max_lead_times_sec 900 1800 3600
--min_link_distances_metres 0 10000 --max_link_distances_metres 10000 100000
"""

import argparse
import numpy
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import link_events_to_storms as events2storms
from gewittergefahr.gg_utils import error_checking

STORM_TO_TORNADOES_FILE_INPUT_ARG = 'input_storm_to_tornadoes_file_name'
MIN_LEAD_TIMES_INPUT_ARG = 'min_lead_times_sec'
MAX_LEAD_TIMES_INPUT_ARG = 'max_lead_times_sec'
MIN_LINK_DISTANCES_INPUT_ARG = 'min_link_distances_metres'
MAX_LINK_DISTANCES_INPUT_ARG = 'max_link_distances_metres'
LABEL_FILE_INPUT_ARG = 'output_label_file_name'

STORM_TO_TORNADOES_FILE_HELP_STRING = (
    'Name of file containing storm-to-tornado linkages.  Should contain columns'
    ' listed in `link_events_to_storms.write_storm_to_tornadoes_table`.')
MIN_LEAD_TIMES_HELP_STRING = (
    'List of minimum lead times (one for each time window).')
MAX_LEAD_TIMES_HELP_STRING = (
    'List of max lead times (one for each time window).')
MIN_LINK_DISTANCES_HELP_STRING = (
    'List of minimum linkage distances (one for each distance window).')
MAX_LINK_DISTANCES_HELP_STRING = (
    'List of max linkage distances (one for each distance window).')
LABEL_FILE_HELP_STRING = (
    'Name of output file (will contain binary label for each storm object, '
    'lead-time window, and distance window; to be written by'
    '`labels.write_tornado_labels`).')

DEFAULT_MIN_LEAD_TIMES_SEC = numpy.array([0, 900, 1800, 2700, 3600, 0])
DEFAULT_MAX_LEAD_TIMES_SEC = numpy.array([900, 1800, 2700, 3600, 5400, 7200])
DEFAULT_MIN_LINK_DISTANCES_METRES = numpy.array([0])
DEFAULT_MAX_LINK_DISTANCES_METRES = numpy.array([100000])

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + STORM_TO_TORNADOES_FILE_INPUT_ARG, type=str, required=True,
    help=STORM_TO_TORNADOES_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_LEAD_TIMES_INPUT_ARG, type=int, nargs='+', required=False,
    default=DEFAULT_MIN_LEAD_TIMES_SEC, help=MIN_LEAD_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LEAD_TIMES_INPUT_ARG, type=int, nargs='+', required=False,
    default=DEFAULT_MAX_LEAD_TIMES_SEC, help=MAX_LEAD_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_LINK_DISTANCES_INPUT_ARG, type=int, nargs='+', required=False,
    default=DEFAULT_MIN_LINK_DISTANCES_METRES,
    help=MIN_LINK_DISTANCES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LINK_DISTANCES_INPUT_ARG, type=int, nargs='+', required=False,
    default=DEFAULT_MAX_LINK_DISTANCES_METRES,
    help=MAX_LINK_DISTANCES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LABEL_FILE_INPUT_ARG, type=str, required=True,
    help=LABEL_FILE_HELP_STRING)


def _label_tornado_occurrence(
        input_storm_to_tornadoes_file_name, min_lead_times_sec,
        max_lead_times_sec, min_link_distances_metres,
        max_link_distances_metres, output_label_file_name):
    """Creates label for each storm object, lead-time window, distance window.

    M = number of lead-time windows
    N = number of distance windows

    :param input_storm_to_tornadoes_file_name: Name of file containing storm-to-
        tornado linkages.  Should contain columns listed in
        `link_events_to_storms.write_storm_to_tornadoes_table`.
    :param min_lead_times_sec: length-M numpy array of minimum lead times.
    :param max_lead_times_sec: length-M numpy array of max lead times.
    :param min_link_distances_metres: length-N numpy array of minimum linkage
        distances.
    :param max_link_distances_metres: length-N numpy array of max linkage
        distances.
    :param output_label_file_name: Name of output file (will contain binary
        label for each storm object, lead-time window, and distance window; to
        be written by `labels.write_tornado_labels`).
    """

    error_checking.assert_is_numpy_array(min_lead_times_sec, num_dimensions=1)
    num_lead_time_windows = len(min_lead_times_sec)
    error_checking.assert_is_numpy_array(
        max_lead_times_sec,
        exact_dimensions=numpy.array([num_lead_time_windows]))

    error_checking.assert_is_numpy_array(
        min_link_distances_metres, num_dimensions=1)
    num_distance_windows = len(min_link_distances_metres)
    error_checking.assert_is_numpy_array(
        max_link_distances_metres,
        exact_dimensions=numpy.array([num_distance_windows]))

    for i in range(num_lead_time_windows):
        for j in range(num_distance_windows):
            labels.check_label_params(
                min_lead_time_sec=min_lead_times_sec[i],
                max_lead_time_sec=max_lead_times_sec[i],
                min_link_distance_metres=min_link_distances_metres[j],
                max_link_distance_metres=max_link_distances_metres[j])

    print 'Reading storm-to-tornado linkages from: "{0:s}"...'.format(
        input_storm_to_tornadoes_file_name)
    storm_to_tornadoes_table = events2storms.read_storm_to_tornadoes_table(
        input_storm_to_tornadoes_file_name)

    for i in range(num_lead_time_windows):
        for j in range(num_distance_windows):
            print (
                'Creating tornado labels for lead time of {0:d}-{1:d} seconds, '
                'linkage distance of {2:.0f}-{3:.0f} metres...').format(
                    min_lead_times_sec[i], max_lead_times_sec[i],
                    min_link_distances_metres[j], max_link_distances_metres[j])

            storm_to_tornadoes_table = labels.label_tornado_occurrence(
                storm_to_tornadoes_table=storm_to_tornadoes_table,
                min_lead_time_sec=min_lead_times_sec[i],
                max_lead_time_sec=max_lead_times_sec[i],
                min_link_distance_metres=min_link_distances_metres[j],
                max_link_distance_metres=max_link_distances_metres[j])

    print 'Writing tornado labels to: "{0:s}"...'.format(output_label_file_name)
    labels.write_tornado_labels(
        storm_to_tornadoes_table, output_label_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    STORM_TO_TORNADOES_FILE_NAME = getattr(
        INPUT_ARG_OBJECT, STORM_TO_TORNADOES_FILE_INPUT_ARG)
    LABEL_FILE_NAME = getattr(INPUT_ARG_OBJECT, LABEL_FILE_INPUT_ARG)

    MIN_LEAD_TIMES_SEC = numpy.array(
        getattr(INPUT_ARG_OBJECT, MIN_LEAD_TIMES_INPUT_ARG), dtype=int)
    MAX_LEAD_TIMES_SEC = numpy.array(
        getattr(INPUT_ARG_OBJECT, MAX_LEAD_TIMES_INPUT_ARG), dtype=int)

    MIN_LINK_DISTANCES_METRES = numpy.array(getattr(
        INPUT_ARG_OBJECT, MIN_LINK_DISTANCES_INPUT_ARG), dtype=float)
    MAX_LINK_DISTANCES_METRES = numpy.array(getattr(
        INPUT_ARG_OBJECT, MAX_LINK_DISTANCES_INPUT_ARG), dtype=float)

    _label_tornado_occurrence(
        input_storm_to_tornadoes_file_name=STORM_TO_TORNADOES_FILE_NAME,
        min_lead_times_sec=MIN_LEAD_TIMES_SEC,
        max_lead_times_sec=MAX_LEAD_TIMES_SEC,
        min_link_distances_metres=MIN_LINK_DISTANCES_METRES,
        max_link_distances_metres=MAX_LINK_DISTANCES_METRES,
        output_label_file_name=LABEL_FILE_NAME)
