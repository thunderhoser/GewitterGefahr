"""Creates hazard label for each storm object, lead-time window, and buffer.

These labels are meant to be used as target variables for machine learning.

--- EXAMPLE ---

Suppose that you want two lead-time windows (0-30 and 30+ minutes) and three
distance buffers (inside storm, 0-10 km outside storm, 10+ km outside storm).
Lead times and buffer distances of 100 000 (seconds and metres, respectively)
are essentially infinite.

To create tornado labels:

create_wind_or_tornado_labels.py --linkage_dir_name="${LINKAGE_DIR_NAME}" \
--spc_date_string="${SPC_DATE_STRING}" --event_type_string="tornado" \
--min_lead_times_sec 0 1800 --max_lead_times_sec 1800 100000 \
--min_link_distances_metres 0 1 10000 \
--max_link_distances_metres 0 10000 100000

To create wind labels (assuming that the 3 classes are 0-30 kt, 30-50 kt, and
50+ kt):

create_wind_or_tornado_labels.py --linkage_dir_name="${LINKAGE_DIR_NAME}" \
--spc_date_string="${SPC_DATE_STRING}" --event_type_string="wind" \
--wind_speed_percentile_level="${WIND_SPEED_PERCENTILE_LEVEL}" \
--class_cutoffs_kt 30 50 \
--min_lead_times_sec 0 1800 --max_lead_times_sec 1800 100000 \
--min_link_distances_metres 0 1 10000 \
--max_link_distances_metres 0 10000 100000
"""

import argparse
import numpy
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import link_events_to_storms as events2storms
from gewittergefahr.gg_utils import error_checking

LINKAGE_DIR_ARG_NAME = 'input_linkage_dir_name'
SPC_DATE_ARG_NAME = 'spc_date_string'
MIN_LEAD_TIMES_ARG_NAME = 'min_lead_times_sec'
MAX_LEAD_TIMES_ARG_NAME = 'max_lead_times_sec'
MIN_LINK_DISTANCES_ARG_NAME = 'min_link_distances_metres'
MAX_LINK_DISTANCES_ARG_NAME = 'max_link_distances_metres'
EVENT_TYPE_ARG_NAME = 'event_type_string'
PERCENTILE_LEVEL_ARG_NAME = 'wind_speed_percentile_level'
CLASS_CUTOFFS_ARG_NAME = 'class_cutoffs_kt'
LABEL_DIR_ARG_NAME = 'output_label_dir_name'

LINKAGE_DIR_HELP_STRING = (
    'Name of top-level directory with linkage files (one per SPC date, readable'
    ' by `link_events_to_storms.read_storm_to_tornadoes_table`).')
SPC_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  Tornado labels '
    'will created for each storm object on this date, each lead-time window, '
    'and each distance buffer.')
MIN_LEAD_TIMES_HELP_STRING = 'List of minimum lead times (one for each window).'
MAX_LEAD_TIMES_HELP_STRING = 'List of max lead times (one for each window).'
MIN_LINK_DISTANCES_HELP_STRING = (
    'List of minimum distances (one for each buffer).')
MAX_LINK_DISTANCES_HELP_STRING = (
    'List of maximum distances (one for each buffer).')
EVENT_TYPE_HELP_STRING = (
    'Labels will be created for this event type.  Must be in the following '
    'list:\n{0:s}').format(events2storms.VALID_EVENT_TYPE_STRINGS)
PERCENTILE_LEVEL_HELP_STRING = (
    'For each storm object, the label will be based on the [q]th-percentile '
    'wind speed of all observations linked to the storm, where q = `{0:s}`.'
).format(PERCENTILE_LEVEL_ARG_NAME)
CLASS_CUTOFFS_HELP_STRING = (
    'List of cutoffs for wind-speed classes.  Units are kt (nautical miles per '
    'hour).  The lowest class will always begin at 0 kt, and the highest class '
    'will begin at infinity, so do not bother with these values.  For example, '
    'if the inputs are 30 and 50, the classes will 0-30, 30-50, and 50+ kt.')
LABEL_DIR_HELP_STRING = (
    'Name of top-level directory for tornado labels.  One file will be created '
    'here, for the whole SPC date, by `labels.write_tornado_labels`.')

DEFAULT_MIN_LEAD_TIMES_SEC = numpy.array(
    [0, 900, 1800, 2700, 3600, 5400, 0], dtype=int)
DEFAULT_MAX_LEAD_TIMES_SEC = numpy.array(
    [900, 1800, 2700, 3600, 5400, 7200, 7200], dtype=int)
DEFAULT_MIN_LINK_DISTANCES_METRES = numpy.array(
    [0, 1, 5000, 10000], dtype=int)
DEFAULT_MAX_LINK_DISTANCES_METRES = numpy.array(
    [0, 5000, 10000, 30000], dtype=int)
DEFAULT_WIND_SPEED_PERCENTILE_LEVEL = 100.
DEFAULT_CLASS_CUTOFFS_KT = numpy.array([50.])

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + LINKAGE_DIR_ARG_NAME, type=str, required=True,
    help=LINKAGE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_LEAD_TIMES_ARG_NAME, type=int, nargs='+', required=False,
    default=DEFAULT_MIN_LEAD_TIMES_SEC, help=MIN_LEAD_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LEAD_TIMES_ARG_NAME, type=int, nargs='+', required=False,
    default=DEFAULT_MAX_LEAD_TIMES_SEC, help=MAX_LEAD_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_LINK_DISTANCES_ARG_NAME, type=int, nargs='+', required=False,
    default=DEFAULT_MIN_LINK_DISTANCES_METRES,
    help=MIN_LINK_DISTANCES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LINK_DISTANCES_ARG_NAME, type=int, nargs='+', required=False,
    default=DEFAULT_MAX_LINK_DISTANCES_METRES,
    help=MAX_LINK_DISTANCES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EVENT_TYPE_ARG_NAME, type=str, required=True,
    help=EVENT_TYPE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PERCENTILE_LEVEL_ARG_NAME, type=str, required=False,
    default=DEFAULT_WIND_SPEED_PERCENTILE_LEVEL,
    help=PERCENTILE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CLASS_CUTOFFS_ARG_NAME, type=float, nargs='+', required=False,
    default=DEFAULT_CLASS_CUTOFFS_KT, help=CLASS_CUTOFFS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LABEL_DIR_ARG_NAME, type=str, required=True,
    help=LABEL_DIR_HELP_STRING)


def _create_labels(
        top_linkage_dir_name, spc_date_string, min_lead_times_sec,
        max_lead_times_sec, min_link_distances_metres,
        max_link_distances_metres, event_type_string,
        wind_speed_percentile_level, class_cutoffs_kt, top_label_dir_name):
    """Creates hazard label for each storm object, lead-time window, and buffer.

    :param top_linkage_dir_name: See documentation at top of file.
    :param spc_date_string: Same.
    :param min_lead_times_sec: Same.
    :param max_lead_times_sec: Same.
    :param min_link_distances_metres: Same.
    :param max_link_distances_metres: Same.
    :param event_type_string: Same.
    :param wind_speed_percentile_level: Same.
    :param class_cutoffs_kt: Same.
    :param top_label_dir_name: Same.
    """

    error_checking.assert_is_numpy_array(min_lead_times_sec, num_dimensions=1)
    num_lead_time_windows = len(min_lead_times_sec)
    error_checking.assert_is_numpy_array(
        max_lead_times_sec,
        exact_dimensions=numpy.array([num_lead_time_windows]))

    error_checking.assert_is_numpy_array(
        min_link_distances_metres, num_dimensions=1)
    num_distance_buffers = len(min_link_distances_metres)
    error_checking.assert_is_numpy_array(
        max_link_distances_metres,
        exact_dimensions=numpy.array([num_distance_buffers]))

    for i in range(num_lead_time_windows):
        for j in range(num_distance_buffers):
            labels.check_label_params(
                min_lead_time_sec=min_lead_times_sec[i],
                max_lead_time_sec=max_lead_times_sec[i],
                min_link_distance_metres=min_link_distances_metres[j],
                max_link_distance_metres=max_link_distances_metres[j])

    linkage_file_name = events2storms.find_storm_to_events_file(
        top_directory_name=top_linkage_dir_name,
        event_type_string=event_type_string, spc_date_string=spc_date_string,
        raise_error_if_missing=True)

    print 'Reading {0:s} linkages from: "{1:s}"...\n'.format(
        event_type_string, linkage_file_name)

    if event_type_string == events2storms.TORNADO_EVENT_TYPE_STRING:
        storm_to_events_table = events2storms.read_storm_to_tornadoes_table(
            linkage_file_name)
    else:
        storm_to_events_table = events2storms.read_storm_to_winds_table(
            linkage_file_name)

    for i in range(num_lead_time_windows):
        for j in range(num_distance_buffers):
            print (
                'Creating {0:s} label for each storm object with {1:d}--{2:d}'
                '-second lead time and {3:d}--{4:d}-metre distance buffer...'
            ).format(event_type_string, min_lead_times_sec[i],
                     max_lead_times_sec[i], min_link_distances_metres[j],
                     max_link_distances_metres[j])

            if event_type_string == events2storms.TORNADO_EVENT_TYPE_STRING:
                storm_to_events_table = labels.label_tornado_occurrence(
                    storm_to_tornadoes_table=storm_to_events_table,
                    min_lead_time_sec=min_lead_times_sec[i],
                    max_lead_time_sec=max_lead_times_sec[i],
                    min_link_distance_metres=min_link_distances_metres[j],
                    max_link_distance_metres=max_link_distances_metres[j])
            else:
                storm_to_events_table = labels.label_wind_speed_for_classification(
                    storm_to_winds_table=storm_to_events_table,
                    min_lead_time_sec=min_lead_times_sec[i],
                    max_lead_time_sec=max_lead_times_sec[i],
                    min_link_distance_metres=min_link_distances_metres[j],
                    max_link_distance_metres=max_link_distances_metres[j],
                    percentile_level=wind_speed_percentile_level,
                    class_cutoffs_kt=class_cutoffs_kt)

    label_file_name = labels.find_label_file(
        top_directory_name=top_label_dir_name,
        event_type_string=event_type_string, spc_date_string=spc_date_string,
        raise_error_if_missing=False)

    print '\nWriting {0:s} labels to: "{1:s}"...'.format(
        event_type_string, label_file_name)

    if event_type_string == events2storms.TORNADO_EVENT_TYPE_STRING:
        labels.write_tornado_labels(
            storm_to_tornadoes_table=storm_to_events_table,
            pickle_file_name=label_file_name)
    else:
        labels.write_wind_speed_labels(
            storm_to_winds_table=storm_to_events_table,
            pickle_file_name=label_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    TOP_LINKAGE_DIR_NAME = getattr(INPUT_ARG_OBJECT, LINKAGE_DIR_ARG_NAME)
    SPC_DATE_STRING = getattr(INPUT_ARG_OBJECT, SPC_DATE_ARG_NAME)

    MIN_LEAD_TIMES_SEC = numpy.array(
        getattr(INPUT_ARG_OBJECT, MIN_LEAD_TIMES_ARG_NAME), dtype=int)
    MAX_LEAD_TIMES_SEC = numpy.array(
        getattr(INPUT_ARG_OBJECT, MAX_LEAD_TIMES_ARG_NAME), dtype=int)

    MIN_LINK_DISTANCES_METRES = numpy.array(getattr(
        INPUT_ARG_OBJECT, MIN_LINK_DISTANCES_ARG_NAME), dtype=float)
    MAX_LINK_DISTANCES_METRES = numpy.array(getattr(
        INPUT_ARG_OBJECT, MAX_LINK_DISTANCES_ARG_NAME), dtype=float)

    EVENT_TYPE_STRING = getattr(INPUT_ARG_OBJECT, EVENT_TYPE_ARG_NAME)
    WIND_SPEED_PERCENTILE_LEVEL = getattr(
        INPUT_ARG_OBJECT, PERCENTILE_LEVEL_ARG_NAME)
    CLASS_CUTOFFS_KT = getattr(INPUT_ARG_OBJECT, CLASS_CUTOFFS_ARG_NAME)

    TOP_LABEL_DIR_NAME = getattr(INPUT_ARG_OBJECT, LABEL_DIR_ARG_NAME)

    _create_labels(
        top_linkage_dir_name=TOP_LINKAGE_DIR_NAME,
        spc_date_string=SPC_DATE_STRING, min_lead_times_sec=MIN_LEAD_TIMES_SEC,
        max_lead_times_sec=MAX_LEAD_TIMES_SEC,
        min_link_distances_metres=MIN_LINK_DISTANCES_METRES,
        max_link_distances_metres=MAX_LINK_DISTANCES_METRES,
        event_type_string=EVENT_TYPE_STRING,
        wind_speed_percentile_level=WIND_SPEED_PERCENTILE_LEVEL,
        class_cutoffs_kt=CLASS_CUTOFFS_KT,
        top_label_dir_name=TOP_LABEL_DIR_NAME)
