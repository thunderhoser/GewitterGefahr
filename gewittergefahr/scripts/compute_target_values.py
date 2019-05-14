"""Computes target value for each storm object, lead-time window, and buffer.

These values are meant to be used as predictands for machine learning.

--- EXAMPLES ---

Suppose that you want two lead-time windows (0-30 and 30+ minutes) and three
distance buffers (inside storm, 0-10 km outside storm, and 10+ km outside
storm).  Lead times and buffer distances of 100 000 (seconds and metres,
respectively) are essentially infinite.

To create tornado labels:

compute_target_values.py --input_linkage_dir_name="${LINKAGE_DIR_NAME}" \
--spc_date_string="${SPC_DATE_STRING}"
--event_type_string="tornado" \
--min_lead_times_sec 0 1800
--max_lead_times_sec 1800 100000 \
--min_link_distances_metres 0 1 10000 \
--max_link_distances_metres 0 10000 100000

To create wind labels (with the three classes being 0-30 kt, 30-50 kt, and
50+ kt):

compute_target_values.py --input_linkage_dir_name="${LINKAGE_DIR_NAME}" \
--spc_date_string="${SPC_DATE_STRING}"
--event_type_string="wind" \
--wind_speed_percentile_level="${WIND_SPEED_PERCENTILE_LEVEL}" \
--wind_speed_cutoffs_kt 30 50 \
--min_lead_times_sec 0 1800
--max_lead_times_sec 1800 100000 \
--min_link_distances_metres 0 1 10000 \
--max_link_distances_metres 0 10000 100000
"""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import general_utils
from gewittergefahr.gg_utils import linkage
from gewittergefahr.gg_utils import target_val_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

LINKAGE_DIR_ARG_NAME = 'input_linkage_dir_name'
FIRST_SPC_DATE_ARG_NAME = 'first_spc_date_string'
LAST_SPC_DATE_ARG_NAME = 'last_spc_date_string'
MIN_LEAD_TIMES_ARG_NAME = 'min_lead_times_sec'
MAX_LEAD_TIMES_ARG_NAME = 'max_lead_times_sec'
MIN_LINK_DISTANCES_ARG_NAME = 'min_link_distances_metres'
MAX_LINK_DISTANCES_ARG_NAME = 'max_link_distances_metres'
EVENT_TYPE_ARG_NAME = 'event_type_string'
PERCENTILE_LEVEL_ARG_NAME = 'wind_speed_percentile_level'
CUTOFFS_ARG_NAME = 'wind_speed_cutoffs_kt'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

LINKAGE_DIR_HELP_STRING = (
    'Name of top-level directory with linkage files.  One file per SPC date '
    'will be found by `linkage.find_linkage_file` and read by '
    '`linkage.read_linkage_file`.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  This script will operate *independently* on'
    ' each day in `{0:s}`...`{1:s}`.'
).format(FIRST_SPC_DATE_ARG_NAME, LAST_SPC_DATE_ARG_NAME)

MIN_LEAD_TIMES_HELP_STRING = 'List of minimum lead times (one for each window).'

MAX_LEAD_TIMES_HELP_STRING = 'List of max lead times (one for each window).'

MIN_LINK_DISTANCES_HELP_STRING = (
    'List of minimum distances (one for each buffer).  0 means "inside storm".')

MAX_LINK_DISTANCES_HELP_STRING = (
    'List of max distances (one for each buffer).  0 means "inside storm".')

EVENT_TYPE_HELP_STRING = (
    'Target variables will be based on this event type.  Valid event types '
    'listed below.\n{0:s}'
).format(linkage.VALID_EVENT_TYPE_STRINGS)

PERCENTILE_LEVEL_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Target variables will be based on this '
    'percentile of wind speeds linked to the storm object.  See '
    '`target_val_utils._check_target_params` for more details.'
).format(EVENT_TYPE_ARG_NAME, linkage.WIND_EVENT_STRING)

CUTOFFS_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Classes will be based on these cutoffs, '
    'applied to the `{2:s}`th percentile of wind speeds linked to the storm '
    'object.  See `target_val_utils._check_target_params` for more details.'
).format(EVENT_TYPE_ARG_NAME, linkage.WIND_EVENT_STRING,
         PERCENTILE_LEVEL_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Files will be written by '
    '`target_val_utils.write_target_values`, to locations therein determined by'
    ' `target_val_utils.find_target_file`.')

DEFAULT_MIN_LEAD_TIMES_SEC = numpy.array(
    [0, 900, 1800, 2700, 3600, 5400, 0], dtype=int
)
DEFAULT_MAX_LEAD_TIMES_SEC = numpy.array(
    [900, 1800, 2700, 3600, 5400, 7200, 7200], dtype=int
)
DEFAULT_MIN_LINK_DISTANCES_METRES = numpy.array(
    [0, 1, 5000, 0], dtype=int
)
DEFAULT_MAX_LINK_DISTANCES_METRES = numpy.array(
    [0, 5000, 10000, 30000], dtype=int
)

DEFAULT_PERCENTILE_LEVEL = 100.
DEFAULT_CUTOFFS_KT = numpy.array([50.])

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + LINKAGE_DIR_ARG_NAME, type=str, required=True,
    help=LINKAGE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_ARG_NAME, type=str, required=True,
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
    '--' + PERCENTILE_LEVEL_ARG_NAME, type=float, required=False,
    default=DEFAULT_PERCENTILE_LEVEL, help=PERCENTILE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CUTOFFS_ARG_NAME, type=float, nargs='+', required=False,
    default=DEFAULT_CUTOFFS_KT, help=CUTOFFS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _compute_targets_one_day(
        storm_to_events_table, spc_date_string, min_lead_times_sec,
        max_lead_times_sec, min_link_distances_metres,
        max_link_distances_metres, event_type_string,
        wind_speed_percentile_level, wind_speed_cutoffs_kt,
        top_output_dir_name):
    """Computes target values for one SPC date.

    :param storm_to_events_table: pandas DataFrame returned by
        `linkage.read_linkage_file`.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param min_lead_times_sec: See documentation at top of file.
    :param max_lead_times_sec: Same.
    :param min_link_distances_metres: Same.
    :param max_link_distances_metres: Same.
    :param event_type_string: Same.
    :param wind_speed_percentile_level: Same.
    :param wind_speed_cutoffs_kt: Same.
    :param top_output_dir_name: Same.
    """

    num_lead_time_windows = len(min_lead_times_sec)
    num_distance_buffers = len(min_link_distances_metres)

    if event_type_string == linkage.WIND_EVENT_STRING:
        list_of_cutoff_arrays_kt = general_utils.split_array_by_nan(
            wind_speed_cutoffs_kt)
        num_cutoff_sets = len(wind_speed_cutoffs_kt)
    else:
        list_of_cutoff_arrays_kt = None
        num_cutoff_sets = 1

    target_names = []

    for i in range(num_lead_time_windows):
        for j in range(num_distance_buffers):
            for k in range(num_cutoff_sets):
                if event_type_string == linkage.WIND_EVENT_STRING:
                    this_target_name = target_val_utils.target_params_to_name(
                        min_lead_time_sec=min_lead_times_sec[i],
                        max_lead_time_sec=max_lead_times_sec[i],
                        min_link_distance_metres=min_link_distances_metres[j],
                        max_link_distance_metres=max_link_distances_metres[j],
                        wind_speed_percentile_level=wind_speed_percentile_level,
                        wind_speed_cutoffs_kt=list_of_cutoff_arrays_kt[k])

                    target_names.append(this_target_name)
                    print (
                        'Computing labels for "{0:s}" on SPC date {1:s}...'
                    ).format(this_target_name, spc_date_string)

                    storm_to_events_table = (
                        target_val_utils.create_wind_classification_targets(
                            storm_to_winds_table=storm_to_events_table,
                            min_lead_time_sec=min_lead_times_sec[i],
                            max_lead_time_sec=max_lead_times_sec[i],
                            min_link_distance_metres=
                            min_link_distances_metres[j],
                            max_link_distance_metres=
                            max_link_distances_metres[j],
                            percentile_level=wind_speed_percentile_level,
                            class_cutoffs_kt=list_of_cutoff_arrays_kt[k])
                    )
                else:
                    this_target_name = target_val_utils.target_params_to_name(
                        min_lead_time_sec=min_lead_times_sec[i],
                        max_lead_time_sec=max_lead_times_sec[i],
                        min_link_distance_metres=min_link_distances_metres[j],
                        max_link_distance_metres=max_link_distances_metres[j])

                    target_names.append(this_target_name)
                    print (
                        'Computing labels for "{0:s}" on SPC date {1:s}...'
                    ).format(this_target_name, spc_date_string)

                    storm_to_events_table = (
                        target_val_utils.create_tornado_targets(
                            storm_to_tornadoes_table=storm_to_events_table,
                            min_lead_time_sec=min_lead_times_sec[i],
                            max_lead_time_sec=max_lead_times_sec[i],
                            min_link_distance_metres=
                            min_link_distances_metres[j],
                            max_link_distance_metres=
                            max_link_distances_metres[j]
                        )
                    )

    target_file_name = target_val_utils.find_target_file(
        top_directory_name=top_output_dir_name,
        event_type_string=event_type_string, spc_date_string=spc_date_string,
        raise_error_if_missing=False)

    print 'Writing target values to: "{0:s}"...'.format(target_file_name)
    target_val_utils.write_target_values(
        storm_to_events_table=storm_to_events_table, target_names=target_names,
        netcdf_file_name=target_file_name)


def _run(top_linkage_dir_name, first_spc_date_string, last_spc_date_string,
         min_lead_times_sec, max_lead_times_sec, min_link_distances_metres,
         max_link_distances_metres, event_type_string,
         wind_speed_percentile_level, wind_speed_cutoffs_kt,
         top_output_dir_name):
    """Computes target value for ea storm object, lead-time window, and buffer.

    This is effectively the main method.

    :param top_linkage_dir_name: See documentation at top of file.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param min_lead_times_sec: Same.
    :param max_lead_times_sec: Same.
    :param min_link_distances_metres: Same.
    :param max_link_distances_metres: Same.
    :param event_type_string: Same.
    :param wind_speed_percentile_level: Same.
    :param wind_speed_cutoffs_kt: Same.
    :param top_output_dir_name: Same.
    """

    num_lead_time_windows = len(min_lead_times_sec)
    these_expected_dim = numpy.array([num_lead_time_windows], dtype=int)
    error_checking.assert_is_numpy_array(
        max_lead_times_sec, exact_dimensions=these_expected_dim)

    num_distance_buffers = len(min_link_distances_metres)
    these_expected_dim = numpy.array([num_distance_buffers], dtype=int)
    error_checking.assert_is_numpy_array(
        max_link_distances_metres, exact_dimensions=these_expected_dim)

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    for this_spc_date_string in spc_date_strings:
        this_linkage_file_name = linkage.find_linkage_file(
            top_directory_name=top_linkage_dir_name,
            event_type_string=event_type_string,
            spc_date_string=this_spc_date_string, raise_error_if_missing=False)

        if not os.path.isfile(this_linkage_file_name):
            continue

        print 'Reading data from: "{0:s}"...'.format(this_linkage_file_name)
        this_storm_to_events_table = linkage.read_linkage_file(
            this_linkage_file_name)

        _compute_targets_one_day(
            storm_to_events_table=this_storm_to_events_table,
            spc_date_string=this_spc_date_string,
            min_lead_times_sec=min_lead_times_sec,
            max_lead_times_sec=max_lead_times_sec,
            min_link_distances_metres=min_link_distances_metres,
            max_link_distances_metres=max_link_distances_metres,
            event_type_string=event_type_string,
            wind_speed_percentile_level=wind_speed_percentile_level,
            wind_speed_cutoffs_kt=wind_speed_cutoffs_kt,
            top_output_dir_name=top_output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_linkage_dir_name=getattr(INPUT_ARG_OBJECT, LINKAGE_DIR_ARG_NAME),
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_SPC_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_ARG_NAME),
        min_lead_times_sec=numpy.array(
            getattr(INPUT_ARG_OBJECT, MIN_LEAD_TIMES_ARG_NAME), dtype=int),
        max_lead_times_sec=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_LEAD_TIMES_ARG_NAME), dtype=int),
        min_link_distances_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, MIN_LINK_DISTANCES_ARG_NAME),
            dtype=float),
        max_link_distances_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_LINK_DISTANCES_ARG_NAME),
            dtype=float),
        event_type_string=getattr(INPUT_ARG_OBJECT, EVENT_TYPE_ARG_NAME),
        wind_speed_percentile_level=getattr(
            INPUT_ARG_OBJECT, PERCENTILE_LEVEL_ARG_NAME),
        wind_speed_cutoffs_kt=numpy.array(
            getattr(INPUT_ARG_OBJECT, CUTOFFS_ARG_NAME), dtype=float),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
