"""Handles target values (predictands) for machine learning."""

import os.path
import numpy
import netCDF4
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_utils import linkage
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import classification_utils as classifn_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
KT_TO_METRES_PER_SECOND = 1.852 / 3.6

INVALID_STORM_INTEGER = -1
DEAD_STORM_INTEGER = -2

LINKAGE_DISTANCE_PRECISION_METRES = 1.
PERCENTILE_LEVEL_PRECISION = 0.1
WIND_SPEED_CUTOFF_PRECISION_KT = 1.

DEFAULT_MIN_LEAD_TIME_SEC = 0
DEFAULT_MAX_LEAD_TIME_SEC = 86400
DEFAULT_MIN_LINK_DISTANCE_METRES = 0.
DEFAULT_MAX_LINK_DISTANCE_METRES = 100000.
DEFAULT_WIND_SPEED_PERCENTILE_LEVEL = 100.
DEFAULT_WIND_SPEED_CUTOFFS_KT = numpy.array([50.])

MIN_LEAD_TIME_KEY = 'min_lead_time_sec'
MAX_LEAD_TIME_KEY = 'max_lead_time_sec'
MIN_LINKAGE_DISTANCE_KEY = 'min_link_distance_metres'
MAX_LINKAGE_DISTANCE_KEY = 'max_link_distance_metres'
PERCENTILE_LEVEL_KEY = 'wind_speed_percentile_level'
WIND_SPEED_CUTOFFS_KEY = 'wind_speed_cutoffs_kt'
EVENT_TYPE_KEY = 'event_type'

REGRESSION_STRING = 'regression'
CLASSIFICATION_STRING = 'classification'
VALID_GOAL_STRINGS = [REGRESSION_STRING, CLASSIFICATION_STRING]

WIND_SPEED_PREFIX_FOR_REGRESSION_NAME = 'wind-speed-m-s01'
WIND_SPEED_PREFIX_FOR_CLASSIFN_NAME = 'wind-speed'
TORNADO_PREFIX = 'tornado'

CHARACTER_DIMENSION_KEY = 'storm_id_character'
STORM_OBJECT_DIMENSION_KEY = 'storm_object'
STORM_IDS_KEY = 'storm_ids'
VALID_TIMES_KEY = 'valid_times_unix_sec'
TARGET_VALUES_KEY = 'target_values'

# TODO(thunderhoser): Add unit tests.


def _check_learning_goal(goal_string):
    """Error-checks learning goal.

    :param goal_string: Learning goal.
    :raises: ValueError: if `goal_string not in VALID_GOAL_STRINGS`.
    """

    error_checking.assert_is_string(goal_string)

    if goal_string not in VALID_GOAL_STRINGS:
        error_string = (
            '\n\n{0:s}\nValid learning goals (listed above) do not include '
            '"{1:s}".'
        ).format(VALID_GOAL_STRINGS, goal_string)

        raise ValueError(error_string)


def _check_target_params(
        min_lead_time_sec, max_lead_time_sec, min_link_distance_metres,
        max_link_distance_metres, wind_speed_percentile_level=None,
        wind_speed_cutoffs_kt=None):
    """Error-checks parameters for target variable.

    :param min_lead_time_sec: Minimum lead time.  For a storm object at time t,
        the target value will be based only on events occurring at times from
        [t + `min_lead_time_sec`, t + `max_lead_time_sec`].
    :param max_lead_time_sec: See above.
    :param min_link_distance_metres: Minimum linkage distance.  For each storm
        cell S, target values will be based only on events occurring at
        distances of `min_link_distance_metres`...`max_link_distance_metres`
        outside the storm cell.  If `min_link_distance_metres` = 0, events
        inside the storm cell will also be permitted.
    :param max_link_distance_metres: See above.
    :param wind_speed_percentile_level: Percentile level for wind speed.  For
        each storm object s, the target value will be based on the [q]th
        percentile, where q = `wind_speed_percentile_level`, of all wind speeds
        linked to s.
    :param wind_speed_cutoffs_kt: 1-D numpy array of cutoffs (knots) for wind-
        speed classification.  The lower bound of the first bin will always be
        0 kt, and the upper bound of the last bin will always be infinity, so
        these values do not need to be included.

    :return: target_param_dict: Dictionary with the following keys.
    target_param_dict['min_link_distance_metres']: Rounded version of input.
    target_param_dict['min_link_distance_metres']: Rounded version of input.
    target_param_dict['wind_speed_percentile_level']: Rounded version of input.
    target_param_dict['wind_speed_cutoffs_kt']: Same as input, but including
        0 kt for lower bound of first bin and infinity for upper bound of last
        bin.
    """

    error_checking.assert_is_integer(min_lead_time_sec)
    error_checking.assert_is_geq(min_lead_time_sec, 0)
    error_checking.assert_is_integer(max_lead_time_sec)
    error_checking.assert_is_geq(max_lead_time_sec, min_lead_time_sec)

    min_link_distance_metres = number_rounding.round_to_nearest(
        min_link_distance_metres, LINKAGE_DISTANCE_PRECISION_METRES)
    max_link_distance_metres = number_rounding.round_to_nearest(
        max_link_distance_metres, LINKAGE_DISTANCE_PRECISION_METRES)

    error_checking.assert_is_geq(min_link_distance_metres, 0.)
    error_checking.assert_is_geq(max_link_distance_metres,
                                 min_link_distance_metres)

    if wind_speed_percentile_level is None:
        wind_speed_cutoffs_kt = None
    else:
        wind_speed_percentile_level = number_rounding.round_to_nearest(
            wind_speed_percentile_level, PERCENTILE_LEVEL_PRECISION)

        error_checking.assert_is_geq(wind_speed_percentile_level, 0.)
        error_checking.assert_is_leq(wind_speed_percentile_level, 100.)

    if wind_speed_cutoffs_kt:
        wind_speed_cutoffs_kt = number_rounding.round_to_nearest(
            wind_speed_cutoffs_kt, WIND_SPEED_CUTOFF_PRECISION_KT)
        wind_speed_cutoffs_kt = classifn_utils.classification_cutoffs_to_ranges(
            wind_speed_cutoffs_kt, non_negative_only=True)[0]

    return {
        MIN_LINKAGE_DISTANCE_KEY: min_link_distance_metres,
        MAX_LINKAGE_DISTANCE_KEY: max_link_distance_metres,
        PERCENTILE_LEVEL_KEY: wind_speed_percentile_level,
        WIND_SPEED_CUTOFFS_KEY: wind_speed_cutoffs_kt
    }


def _find_storms_near_end_of_period(storm_to_events_table, max_lead_time_sec):
    """Finds storm objects near end of tracking period.

    This is important because, for storm objects near the end of the tracking
    period -- i.e., for storm objects within `max_lead_time_sec` of the end of
    the period -- we cannot confidently say that an event did not occur.  For
    example, we cannot say "The storm produced no tornadoes within the lead-time
    window."

    :param storm_to_events_table: See doc for `linkage.read_linkage_file`.
    :param max_lead_time_sec: See doc for `_check_target_params`.
    :return: bad_indices: 1-D numpy array with indices of bad storm objects
        (near end of tracking period).  These are row indices into
        `storm_to_events_table`.
    """

    times_before_end_sec = (
        storm_to_events_table[tracking_utils.TRACKING_END_TIME_COLUMN] -
        storm_to_events_table[tracking_utils.TIME_COLUMN]
    )

    return numpy.where(times_before_end_sec < max_lead_time_sec)[0]


def _find_dead_storms(storm_to_events_table, min_lead_time_sec):
    """Finds "dead storms" (those that do not persist beyond minimum lead time).

    :param storm_to_events_table: See doc for `linkage.read_linkage_file`.
    :param min_lead_time_sec: See doc for `_check_target_params`.
    :return: bad_indices: 1-D numpy array with indices of dead storms.  These
        are row indices into `storm_to_events_table`.
    """

    remaining_lifetimes_sec = (
        storm_to_events_table[tracking_utils.CELL_END_TIME_COLUMN] -
        storm_to_events_table[tracking_utils.TIME_COLUMN]
    )

    return numpy.where(remaining_lifetimes_sec < min_lead_time_sec)[0]


def target_params_to_name(
        min_lead_time_sec, max_lead_time_sec, min_link_distance_metres,
        max_link_distance_metres, wind_speed_percentile_level=None,
        wind_speed_cutoffs_kt=None):
    """Creates name for target variable.

    :param min_lead_time_sec: See doc for `_check_target_params`.
    :param max_lead_time_sec: Same.
    :param min_link_distance_metres: Same.
    :param max_link_distance_metres: Same.
    :param wind_speed_percentile_level: Same.
    :param wind_speed_cutoffs_kt: Same.
    :return: target_name: Name of target variable.
    """

    target_param_dict = _check_target_params(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres,
        wind_speed_percentile_level=wind_speed_percentile_level,
        wind_speed_cutoffs_kt=wind_speed_cutoffs_kt)

    min_link_distance_metres = target_param_dict[MIN_LINKAGE_DISTANCE_KEY]
    max_link_distance_metres = target_param_dict[MAX_LINKAGE_DISTANCE_KEY]
    wind_speed_percentile_level = target_param_dict[PERCENTILE_LEVEL_KEY]
    wind_speed_cutoffs_kt = target_param_dict[WIND_SPEED_CUTOFFS_KEY]

    if wind_speed_percentile_level is None:
        target_name = TORNADO_PREFIX + ''
    else:
        if wind_speed_cutoffs_kt is None:
            target_name = WIND_SPEED_PREFIX_FOR_REGRESSION_NAME + ''
        else:
            target_name = WIND_SPEED_PREFIX_FOR_CLASSIFN_NAME + ''

        target_name += '_percentile={0:05.1f}'.format(
            wind_speed_percentile_level)

    target_name += (
        '_lead-time={0:04d}-{1:04d}sec_distance={2:05d}-{3:05d}m'
    ).format(
        min_lead_time_sec, max_lead_time_sec, int(min_link_distance_metres),
        int(max_link_distance_metres)
    )

    if wind_speed_cutoffs_kt is not None:
        cutoff_string = '-'.join(
            ['{0:02d}'.format(int(c)) for c in wind_speed_cutoffs_kt]
        )

        target_name += '_cutoffs={0:s}kt'.format(cutoff_string)

    return target_name


def target_name_to_params(target_name):
    """Parses parameters of target variable from its name.

    :param target_name: Name of target variable.
    :return: target_param_dict: Dictionary with the following keys.
    target_param_dict['min_lead_time_sec']: See doc for `_check_target_params`.
    target_param_dict['max_lead_time_sec']: Same.
    target_param_dict['min_link_distance_metres']: Same.
    target_param_dict['min_link_distance_metres']: Same.
    target_param_dict['wind_speed_percentile_level']: Same.
    target_param_dict['wind_speed_cutoffs_kt']: Same.
    target_param_dict['event_type_string']: Event type (one of the strings
        accepted by `linkage.check_event_type`).
    """

    target_name_parts = target_name.split('_')

    # Determine event type (wind or tornado) and goal (classifn or regression).
    if target_name_parts[0] == WIND_SPEED_PREFIX_FOR_REGRESSION_NAME:
        goal_string = REGRESSION_STRING
        event_type_string = linkage.WIND_EVENT_STRING
        if len(target_name_parts) != 4:
            return None

    elif target_name_parts[0] == WIND_SPEED_PREFIX_FOR_CLASSIFN_NAME:
        goal_string = CLASSIFICATION_STRING
        event_type_string = linkage.WIND_EVENT_STRING
        if len(target_name_parts) != 5:
            return None

    elif target_name_parts[0] == TORNADO_PREFIX:
        goal_string = CLASSIFICATION_STRING
        event_type_string = linkage.TORNADO_EVENT_STRING
        if len(target_name_parts) != 3:
            return None

    else:
        return None

    # Determine percentile level for wind speed.
    wind_speed_percentile_level = None
    if event_type_string == linkage.WIND_EVENT_STRING:
        if not target_name_parts[1].startswith('percentile='):
            return None

        try:
            wind_speed_percentile_level = float(
                target_name_parts[1].replace('percentile=', ''))
        except ValueError:
            return None

        target_name_parts.remove(target_name_parts[1])

    # Determine min/max lead times.
    if not target_name_parts[1].startswith('lead-time='):
        return None

    if not target_name_parts[1].endswith('sec='):
        return None

    lead_time_parts = target_name_parts[1].replace(
        'lead-time=', '').replace('sec', '').split('-')
    if len(lead_time_parts) != 2:
        return None

    try:
        min_lead_time_sec = int(lead_time_parts[0])
        max_lead_time_sec = int(lead_time_parts[1])
    except ValueError:
        return None

    # Determine min/max linkage distances.
    if not target_name_parts[2].startswith('distance='):
        return None

    if not target_name_parts[2].endswith('m'):
        return None

    distance_parts = target_name_parts[2].replace(
        'distance=', '').replace('m', '').split('-')
    if len(distance_parts) != 2:
        return None

    try:
        min_link_distance_metres = float(int(distance_parts[0]))
        max_link_distance_metres = float(int(distance_parts[1]))
    except ValueError:
        return None

    target_param_dict = {
        MIN_LEAD_TIME_KEY: min_lead_time_sec,
        MAX_LEAD_TIME_KEY: max_lead_time_sec,
        MIN_LINKAGE_DISTANCE_KEY: min_link_distance_metres,
        MAX_LINKAGE_DISTANCE_KEY: max_link_distance_metres,
        PERCENTILE_LEVEL_KEY: wind_speed_percentile_level,
        WIND_SPEED_CUTOFFS_KEY: None,
        EVENT_TYPE_KEY: event_type_string
    }

    if not (event_type_string == linkage.WIND_EVENT_STRING and
            goal_string == CLASSIFICATION_STRING):
        return target_param_dict

    if not target_name_parts[3].startswith('cutoffs='):
        return None

    if not target_name_parts[3].endswith('kt'):
        return None

    cutoff_parts = target_name_parts[3].replace(
        'cutoffs=', '').replace('kt', '').split('-')

    try:
        target_param_dict[WIND_SPEED_CUTOFFS_KEY] = numpy.array(
            [int(c) for c in cutoff_parts]
        ).astype(float)
    except ValueError:
        return None

    return target_param_dict


def target_name_to_num_classes(target_name, include_dead_storms=False):
    """Parses number of classes from name of (classifn-based) target variable.

    :param target_name: Name of target variable.
    :param include_dead_storms: Boolean flag.  If True, number of classes will
        include "dead storms" (defined in documentation for
        `_find_dead_storms`).
    :return: num_classes: Number of classes.  If target variable is regression-
        based, will return None.
    """

    target_param_dict = target_name_to_params(target_name)
    if target_param_dict[EVENT_TYPE_KEY] == linkage.TORNADO_EVENT_STRING:
        return 2

    error_checking.assert_is_boolean(include_dead_storms)
    wind_speed_cutoffs_kt = target_param_dict[WIND_SPEED_CUTOFFS_KEY]
    if wind_speed_cutoffs_kt is None:
        return None

    if target_param_dict[MIN_LEAD_TIME_KEY] <= 0:
        return len(wind_speed_cutoffs_kt) + 1

    return len(wind_speed_cutoffs_kt) + 1 + int(include_dead_storms)


def create_wind_regression_targets(
        storm_to_winds_table, min_lead_time_sec=DEFAULT_MIN_LEAD_TIME_SEC,
        max_lead_time_sec=DEFAULT_MAX_LEAD_TIME_SEC,
        min_link_distance_metres=DEFAULT_MIN_LINK_DISTANCE_METRES,
        max_link_distance_metres=DEFAULT_MAX_LINK_DISTANCE_METRES,
        percentile_level=DEFAULT_WIND_SPEED_PERCENTILE_LEVEL):
    """For each storm object, creates regression target based on wind speed.

    :param storm_to_winds_table: See doc for `linkage.read_linkage_file`.
    :param min_lead_time_sec: See doc for `_check_target_params`.
    :param max_lead_time_sec: Same.
    :param min_link_distance_metres: Same.
    :param max_link_distance_metres: Same.
    :param percentile_level: Same.
    :return: storm_to_winds_table: Same as input, but with additional column
        containing target values.  The name of this column is determined by
        `target_params_to_name`.
    """

    target_param_dict = _check_target_params(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres,
        wind_speed_percentile_level=percentile_level,
        wind_speed_cutoffs_kt=None)

    min_link_distance_metres = target_param_dict[MIN_LINKAGE_DISTANCE_KEY]
    max_link_distance_metres = target_param_dict[MAX_LINKAGE_DISTANCE_KEY]
    percentile_level = target_param_dict[PERCENTILE_LEVEL_KEY]

    end_of_period_indices = _find_storms_near_end_of_period(
        storm_to_events_table=storm_to_winds_table,
        max_lead_time_sec=max_lead_time_sec)

    dead_storm_indices = _find_dead_storms(
        storm_to_events_table=storm_to_winds_table,
        min_lead_time_sec=min_lead_time_sec)

    num_storm_objects = len(storm_to_winds_table.index)
    labels_m_s01 = numpy.full(num_storm_objects, numpy.nan)
    labels_m_s01[end_of_period_indices] = INVALID_STORM_INTEGER
    labels_m_s01[dead_storm_indices] = DEAD_STORM_INTEGER

    for i in range(num_storm_objects):
        if i in end_of_period_indices or i in dead_storm_indices:
            continue

        these_relative_times_sec = storm_to_winds_table[
            linkage.RELATIVE_EVENT_TIMES_COLUMN].values[i]
        these_link_distances_metres = storm_to_winds_table[
            linkage.LINKAGE_DISTANCES_COLUMN].values[i]

        these_good_time_flags = numpy.logical_and(
            these_relative_times_sec >= min_lead_time_sec,
            these_relative_times_sec <= max_lead_time_sec)
        these_good_distance_flags = numpy.logical_and(
            these_link_distances_metres >= min_link_distance_metres,
            these_link_distances_metres <= max_link_distance_metres)
        these_good_ob_flags = numpy.logical_and(
            these_good_time_flags, these_good_distance_flags)

        if not numpy.any(these_good_ob_flags):
            # labels_m_s01[i] = 0.
            labels_m_s01[i] = INVALID_STORM_INTEGER
            continue

        these_good_ob_indices = numpy.where(these_good_ob_flags)[0]
        these_wind_speeds_m_s01 = numpy.sqrt(
            storm_to_winds_table[linkage.U_WINDS_COLUMN].values[i][
                these_good_ob_indices] ** 2 +
            storm_to_winds_table[linkage.V_WINDS_COLUMN].values[i][
                these_good_ob_indices] ** 2
        )

        labels_m_s01[i] = numpy.percentile(
            these_wind_speeds_m_s01, percentile_level)

    target_name = target_params_to_name(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres,
        wind_speed_percentile_level=percentile_level,
        wind_speed_cutoffs_kt=None)

    return storm_to_winds_table.assign(**{target_name: labels_m_s01})


def create_wind_classification_targets(
        storm_to_winds_table, min_lead_time_sec=DEFAULT_MIN_LEAD_TIME_SEC,
        max_lead_time_sec=DEFAULT_MAX_LEAD_TIME_SEC,
        min_link_distance_metres=DEFAULT_MIN_LINK_DISTANCE_METRES,
        max_link_distance_metres=DEFAULT_MAX_LINK_DISTANCE_METRES,
        percentile_level=DEFAULT_WIND_SPEED_PERCENTILE_LEVEL,
        class_cutoffs_kt=DEFAULT_WIND_SPEED_CUTOFFS_KT):
    """For each storm object, creates classification target based on wind speed.

    :param storm_to_winds_table: See doc for `linkage.read_linkage_file`.
    :param min_lead_time_sec: See doc for `_check_target_params`.
    :param max_lead_time_sec: Same.
    :param min_link_distance_metres: Same.
    :param max_link_distance_metres: Same.
    :param percentile_level: Same.
    :param class_cutoffs_kt: Same.
    :return: storm_to_winds_table: Same as input, but with additional column
        containing target values.  The name of this column is determined by
        `target_params_to_name`.
    """

    target_param_dict = _check_target_params(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres,
        wind_speed_percentile_level=percentile_level,
        wind_speed_cutoffs_kt=class_cutoffs_kt)

    class_cutoffs_kt = target_param_dict[WIND_SPEED_CUTOFFS_KEY]
    class_cutoffs_m_s01 = class_cutoffs_kt * KT_TO_METRES_PER_SECOND

    storm_to_winds_table = create_wind_regression_targets(
        storm_to_winds_table=storm_to_winds_table,
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres,
        percentile_level=percentile_level)

    target_name = target_params_to_name(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres,
        wind_speed_percentile_level=percentile_level,
        wind_speed_cutoffs_kt=None)

    regression_labels_m_s01 = storm_to_winds_table[target_name].values

    invalid_storm_indices = numpy.where(
        regression_labels_m_s01 == INVALID_STORM_INTEGER
    )[0]
    dead_storm_indices = numpy.where(
        regression_labels_m_s01 == DEAD_STORM_INTEGER
    )[0]

    regression_labels_m_s01[invalid_storm_indices] = 0.
    regression_labels_m_s01[dead_storm_indices] = 0.

    storm_classes = classifn_utils.classify_values(
        input_values=regression_labels_m_s01, class_cutoffs=class_cutoffs_m_s01,
        non_negative_only=True)

    storm_classes[invalid_storm_indices] = INVALID_STORM_INTEGER
    storm_classes[dead_storm_indices] = DEAD_STORM_INTEGER

    target_name = target_params_to_name(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres,
        wind_speed_percentile_level=percentile_level,
        wind_speed_cutoffs_kt=class_cutoffs_kt)

    return storm_to_winds_table.assign(**{target_name: storm_classes})


def create_tornado_targets(
        storm_to_tornadoes_table, min_lead_time_sec=DEFAULT_MIN_LEAD_TIME_SEC,
        max_lead_time_sec=DEFAULT_MAX_LEAD_TIME_SEC,
        min_link_distance_metres=DEFAULT_MIN_LINK_DISTANCE_METRES,
        max_link_distance_metres=DEFAULT_MAX_LINK_DISTANCE_METRES):
    """For each storm object, creates target based on tornado occurrence.

    :param storm_to_tornadoes_table: See doc for `linkage.read_linkage_file`.
    :param min_lead_time_sec: See doc for `_check_target_params`.
    :param max_lead_time_sec: Same.
    :param min_link_distance_metres: Same.
    :param max_link_distance_metres: Same.
    :return: storm_to_tornadoes_table: Same as input, but with additional column
        containing target values.  The name of this column is determined by
        `target_params_to_name`.
    """

    target_param_dict = _check_target_params(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres)

    min_link_distance_metres = target_param_dict[MIN_LINKAGE_DISTANCE_KEY]
    max_link_distance_metres = target_param_dict[MAX_LINKAGE_DISTANCE_KEY]

    end_of_period_indices = _find_storms_near_end_of_period(
        storm_to_events_table=storm_to_tornadoes_table,
        max_lead_time_sec=max_lead_time_sec)

    num_storm_objects = len(storm_to_tornadoes_table.index)
    tornado_classes = numpy.full(num_storm_objects, -1, dtype=int)

    for i in range(num_storm_objects):
        these_relative_times_sec = storm_to_tornadoes_table[
            linkage.RELATIVE_EVENT_TIMES_COLUMN].values[i]
        these_link_distances_metres = storm_to_tornadoes_table[
            linkage.LINKAGE_DISTANCES_COLUMN].values[i]

        these_good_time_flags = numpy.logical_and(
            these_relative_times_sec >= min_lead_time_sec,
            these_relative_times_sec <= max_lead_time_sec)
        these_good_distance_flags = numpy.logical_and(
            these_link_distances_metres >= min_link_distance_metres,
            these_link_distances_metres <= max_link_distance_metres)
        these_good_ob_flags = numpy.logical_and(
            these_good_time_flags, these_good_distance_flags)

        tornado_classes[i] = numpy.any(these_good_ob_flags)
        if i in end_of_period_indices and tornado_classes[i] == 0:
            tornado_classes[i] = INVALID_STORM_INTEGER

    target_name = target_params_to_name(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres)

    return storm_to_tornadoes_table.assign(**{target_name: tornado_classes})


def find_target_file(top_directory_name, event_type_string, spc_date_string,
                     raise_error_if_missing=True, unix_time_sec=None):
    """Locates file with target values for either one time or one SPC date.

    :param top_directory_name: Name of top-level directory with target files.
    :param event_type_string: Event type (must be accepted by
        `linkage.check_event_type`).
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :param unix_time_sec: Valid time.
    :return: target_file_name: Path to linkage file.  If file is missing and
        `raise_error_if_missing = False`, this will be the *expected* path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(top_directory_name)
    linkage.check_event_type(event_type_string)
    error_checking.assert_is_boolean(raise_error_if_missing)

    if unix_time_sec is None:
        time_conversion.spc_date_string_to_unix_sec(spc_date_string)

        if event_type_string == linkage.WIND_EVENT_STRING:
            target_file_name = '{0:s}/{1:s}/wind_labels_{2:s}.nc'.format(
                top_directory_name, spc_date_string[:4], spc_date_string)
        else:
            target_file_name = '{0:s}/{1:s}/tornado_labels_{2:s}.nc'.format(
                top_directory_name, spc_date_string[:4], spc_date_string)
    else:
        spc_date_string = time_conversion.time_to_spc_date_string(unix_time_sec)
        valid_time_string = time_conversion.unix_sec_to_string(
            unix_time_sec, TIME_FORMAT)

        if event_type_string == linkage.WIND_EVENT_STRING:
            target_file_name = '{0:s}/{1:s}/{2:s}/wind_labels_{3:s}.nc'.format(
                top_directory_name, spc_date_string[:4], spc_date_string,
                valid_time_string)
        else:
            target_file_name = (
                '{0:s}/{1:s}/{2:s}/tornado_labels_{3:s}.nc'
            ).format(top_directory_name, spc_date_string[:4], spc_date_string,
                     valid_time_string)

    if raise_error_if_missing and not os.path.isfile(target_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            target_file_name)
        raise ValueError(error_string)

    return target_file_name


def write_target_values(storm_to_events_table, target_names, netcdf_file_name):
    """Writes target values to NetCDF file.

    :param storm_to_events_table: pandas DataFrame created by
        `create_wind_regression_targets`, `create_wind_classification_targets`,
        or `create_tornado_targets`.
    :param target_names: 1-D list with names of target variables to write.  Each
        name must be a column in `storm_to_events_table`.
    :param netcdf_file_name: Path to output file.
    :raises: ValueError: if any item in `target_names` is not a valid name.
    """

    error_checking.assert_is_string_list(target_names)
    error_checking.assert_is_numpy_array(
        numpy.array(target_names), num_dimensions=1)

    for this_target_name in target_names:
        this_param_dict = target_name_to_params(this_target_name)
        if this_param_dict is not None:
            continue

        error_string = (
            '"{0:s}" is not a valid name for a target variable.'
        ).format(this_target_name)
        raise ValueError(error_string)

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    netcdf_dataset = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    storm_ids = storm_to_events_table[tracking_utils.STORM_ID_COLUMN].values

    num_storm_objects = len(storm_ids)
    num_storm_id_chars = 0
    for i in range(num_storm_objects):
        num_storm_id_chars = max([num_storm_id_chars, len(storm_ids[i])])

    netcdf_dataset.createDimension(
        STORM_OBJECT_DIMENSION_KEY, num_storm_objects)
    netcdf_dataset.createDimension(CHARACTER_DIMENSION_KEY, num_storm_id_chars)

    netcdf_dataset.createVariable(
        STORM_IDS_KEY, datatype='S1',
        dimensions=(STORM_OBJECT_DIMENSION_KEY, CHARACTER_DIMENSION_KEY))

    string_type = 'S{0:d}'.format(num_storm_id_chars)
    storm_ids_as_char_array = netCDF4.stringtochar(numpy.array(
        storm_ids, dtype=string_type))
    netcdf_dataset.variables[STORM_IDS_KEY][:] = numpy.array(
        storm_ids_as_char_array)

    netcdf_dataset.createVariable(
        VALID_TIMES_KEY, datatype=numpy.int32,
        dimensions=STORM_OBJECT_DIMENSION_KEY)
    netcdf_dataset.variables[VALID_TIMES_KEY][:] = storm_to_events_table[
        tracking_utils.TIME_COLUMN].values

    for this_target_name in target_names:
        netcdf_dataset.createVariable(
            this_target_name, datatype=numpy.float32,
            dimensions=STORM_OBJECT_DIMENSION_KEY)
        netcdf_dataset.variables[this_target_name][:] = storm_to_events_table[
            this_target_name].values

    netcdf_dataset.close()


def read_target_values(netcdf_file_name, target_name):
    """Reads target values from NetCDF file.

    N = number of storm objects

    :param netcdf_file_name: Path to input file.
    :param target_name: Name of target variable.
    :return: storm_label_dict: Dictionary with the following keys.
    storm_label_dict['storm_ids']: length-N list of storm IDs.
    storm_label_dict['valid_times_unix_sec']: length-N numpy array of valid
        times.
    storm_label_dict['target_values']: length-N numpy array with values of
        `target_name`.
    """

    error_checking.assert_is_string(target_name)
    netcdf_dataset = netcdf_io.open_netcdf(
        netcdf_file_name=netcdf_file_name, raise_error_if_fails=True)

    storm_ids = netCDF4.chartostring(netcdf_dataset.variables[STORM_IDS_KEY][:])
    valid_times_unix_sec = numpy.array(
        netcdf_dataset.variables[VALID_TIMES_KEY][:], dtype=int)
    target_values = numpy.array(
        netcdf_dataset.variables[target_name][:], dtype=int)

    netcdf_dataset.close()

    return {
        STORM_IDS_KEY: [str(s) for s in storm_ids],
        VALID_TIMES_KEY: valid_times_unix_sec,
        TARGET_VALUES_KEY: target_values
    }
