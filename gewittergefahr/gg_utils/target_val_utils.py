"""Handles target values (predictands) for machine learning."""

import os.path
import numpy
import netCDF4
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_io import tornado_io
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

DISTANCE_PRECISION_METRES = 1.
PERCENTILE_LEVEL_PRECISION = 0.1
WIND_SPEED_PRECISION_KT = 1.

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
MIN_FUJITA_RATING_KEY = 'min_fujita_rating'
PERCENTILE_LEVEL_KEY = 'wind_speed_percentile_level'
WIND_SPEED_CUTOFFS_KEY = 'wind_speed_cutoffs_kt'
EVENT_TYPE_KEY = 'event_type'

REGRESSION_STRING = 'regression'
CLASSIFICATION_STRING = 'classification'
VALID_GOAL_STRINGS = [REGRESSION_STRING, CLASSIFICATION_STRING]

WIND_SPEED_REGRESSION_PREFIX = 'wind-speed-m-s01'
WIND_SPEED_CLASSIFN_PREFIX = 'wind-speed'
TORNADO_PREFIX = 'tornado'
TORNADOGENESIS_PREFIX = 'tornadogenesis'

CHARACTER_DIMENSION_KEY = 'storm_id_character'
STORM_OBJECT_DIMENSION_KEY = 'storm_object'

FULL_IDS_KEY = 'full_id_strings'
VALID_TIMES_KEY = 'valid_times_unix_sec'
TARGET_NAMES_KEY = 'target_names'
TARGET_MATRIX_KEY = 'target_matrix'


def _check_learning_goal(goal_string):
    """Error-checks learning goal.

    :param goal_string: Learning goal.
    :raises: ValueError: if `goal_string not in VALID_GOAL_STRINGS`.
    """

    error_checking.assert_is_string(goal_string)

    if goal_string not in VALID_GOAL_STRINGS:
        error_string = (
            '\n{0:s}\nValid learning goals (listed above) do not include '
            '"{1:s}".'
        ).format(str(VALID_GOAL_STRINGS), goal_string)

        raise ValueError(error_string)


def _check_target_params(
        min_lead_time_sec, max_lead_time_sec, min_link_distance_metres,
        max_link_distance_metres, tornadogenesis_only=None, min_fujita_rating=0,
        wind_speed_percentile_level=None, wind_speed_cutoffs_kt=None):
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
    :param tornadogenesis_only: Boolean flag.  If True, labels are for
        tornadogenesis.  If False, labels are for occurrence (pre-existing
        tornado or genesis).  If None, labels are for wind speed.
    :param min_fujita_rating: [used if `tornadogenesis_only == True`]
        Minimum Fujita rating (integer).
    :param wind_speed_percentile_level: [used if `tornadogenesis_only == False`]
        Percentile level for wind speed.  For each storm object s, the target
        value will be based on the [q]th percentile, where
        q = `wind_speed_percentile_level`, of all wind speeds linked to s.
    :param wind_speed_cutoffs_kt: [used if `tornadogenesis_only == False`]
        1-D numpy array of cutoffs (knots) for wind-speed classification.  The
        lower bound of the first bin will always be 0 kt, and the upper bound of
        the last bin will always be infinity, so these values do not need to be
        included.

    :return: target_param_dict: Dictionary with the following keys.
    target_param_dict['min_link_distance_metres']: Rounded version of input.
    target_param_dict['min_link_distance_metres']: Rounded version of input.
    target_param_dict['min_fujita_rating']: See input.
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
        min_link_distance_metres, DISTANCE_PRECISION_METRES
    )
    max_link_distance_metres = number_rounding.round_to_nearest(
        max_link_distance_metres, DISTANCE_PRECISION_METRES
    )

    error_checking.assert_is_geq(min_link_distance_metres, 0.)
    error_checking.assert_is_geq(
        max_link_distance_metres, min_link_distance_metres
    )

    if tornadogenesis_only is not None:
        error_checking.assert_is_boolean(tornadogenesis_only)
        error_checking.assert_is_integer(min_fujita_rating)
        error_checking.assert_is_geq(min_fujita_rating, 0)
        error_checking.assert_is_leq(min_fujita_rating, 5)

        return {
            MIN_LINKAGE_DISTANCE_KEY: min_link_distance_metres,
            MAX_LINKAGE_DISTANCE_KEY: max_link_distance_metres,
            MIN_FUJITA_RATING_KEY: min_fujita_rating,
            PERCENTILE_LEVEL_KEY: None,
            WIND_SPEED_CUTOFFS_KEY: None
        }

    wind_speed_percentile_level = number_rounding.round_to_nearest(
        wind_speed_percentile_level, PERCENTILE_LEVEL_PRECISION
    )
    error_checking.assert_is_geq(wind_speed_percentile_level, 0.)
    error_checking.assert_is_leq(wind_speed_percentile_level, 100.)

    if wind_speed_cutoffs_kt is not None:
        wind_speed_cutoffs_kt = number_rounding.round_to_nearest(
            wind_speed_cutoffs_kt, WIND_SPEED_PRECISION_KT
        )
        wind_speed_cutoffs_kt = classifn_utils.classification_cutoffs_to_ranges(
            wind_speed_cutoffs_kt, non_negative_only=True
        )[0]

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
        storm_to_events_table[tracking_utils.TRACKING_END_TIME_COLUMN].values -
        storm_to_events_table[tracking_utils.VALID_TIME_COLUMN].values
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
        storm_to_events_table[tracking_utils.CELL_END_TIME_COLUMN].values -
        storm_to_events_table[tracking_utils.VALID_TIME_COLUMN].values
    )

    return numpy.where(remaining_lifetimes_sec < min_lead_time_sec)[0]


def target_params_to_name(
        min_lead_time_sec, max_lead_time_sec, min_link_distance_metres,
        max_link_distance_metres, tornadogenesis_only=None, min_fujita_rating=0,
        wind_speed_percentile_level=None, wind_speed_cutoffs_kt=None):
    """Creates name for target variable.

    :param min_lead_time_sec: See doc for `_check_target_params`.
    :param max_lead_time_sec: Same.
    :param min_link_distance_metres: Same.
    :param max_link_distance_metres: Same.
    :param tornadogenesis_only: Same.
    :param min_fujita_rating: Same.
    :param wind_speed_percentile_level: Same.
    :param wind_speed_cutoffs_kt: Same.
    :return: target_name: Name of target variable.
    """

    target_param_dict = _check_target_params(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres,
        tornadogenesis_only=tornadogenesis_only,
        min_fujita_rating=min_fujita_rating,
        wind_speed_percentile_level=wind_speed_percentile_level,
        wind_speed_cutoffs_kt=wind_speed_cutoffs_kt)

    min_link_distance_metres = target_param_dict[MIN_LINKAGE_DISTANCE_KEY]
    max_link_distance_metres = target_param_dict[MAX_LINKAGE_DISTANCE_KEY]
    wind_speed_percentile_level = target_param_dict[PERCENTILE_LEVEL_KEY]
    wind_speed_cutoffs_kt = target_param_dict[WIND_SPEED_CUTOFFS_KEY]

    if tornadogenesis_only is None:
        target_name = (
            WIND_SPEED_REGRESSION_PREFIX if wind_speed_cutoffs_kt is None
            else WIND_SPEED_CLASSIFN_PREFIX
        )

        target_name += '_percentile={0:05.1f}'.format(
            wind_speed_percentile_level
        )
    else:
        target_name = (
            TORNADOGENESIS_PREFIX if tornadogenesis_only else TORNADO_PREFIX
        )

    target_name += (
        '_lead-time={0:04d}-{1:04d}sec_distance={2:05d}-{3:05d}m'
    ).format(
        min_lead_time_sec, max_lead_time_sec,
        int(min_link_distance_metres), int(max_link_distance_metres)
    )

    if tornadogenesis_only is not None:
        target_name += '_min-fujita={0:d}'.format(min_fujita_rating)

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
    target_param_dict['max_link_distance_metres']: Same.
    target_param_dict['min_fujita_rating']: Same.
    target_param_dict['wind_speed_percentile_level']: Same.
    target_param_dict['wind_speed_cutoffs_kt']: Same.
    target_param_dict['event_type_string']: Event type (one of the strings
        accepted by `linkage.check_event_type`).
    """

    words = target_name.split('_')

    # Determine event type and learning goal (regression or classification).
    if words[0] == WIND_SPEED_REGRESSION_PREFIX:
        goal_string = REGRESSION_STRING
        event_type_string = linkage.WIND_EVENT_STRING

        if len(words) != 4:
            return None

    elif words[0] == WIND_SPEED_CLASSIFN_PREFIX:
        goal_string = CLASSIFICATION_STRING
        event_type_string = linkage.WIND_EVENT_STRING

        if len(words) != 5:
            return None

    elif words[0] == TORNADO_PREFIX:
        goal_string = CLASSIFICATION_STRING
        event_type_string = linkage.TORNADO_EVENT_STRING
    elif words[0] == TORNADOGENESIS_PREFIX:
        goal_string = CLASSIFICATION_STRING
        event_type_string = linkage.TORNADOGENESIS_EVENT_STRING
    else:
        return None

    tornado = event_type_string in [
        linkage.TORNADO_EVENT_STRING, linkage.TORNADOGENESIS_EVENT_STRING
    ]
    min_fujita_rating = 0

    if tornado and len(words) == 4:
        these_subwords = words[-1].split('=')
        if these_subwords[0] != 'min-fujita':
            return None

        try:
            min_fujita_rating = int(these_subwords[1])
        except ValueError:
            return None

        words = words[:-1]

    if tornado and len(words) != 3:
        return None

    # Determine percentile level for wind speed.
    wind_speed_percentile_level = None

    if event_type_string == linkage.WIND_EVENT_STRING:
        these_subwords = words[1].split('=')
        if these_subwords[0] != 'percentile':
            return None

        try:
            wind_speed_percentile_level = float(these_subwords[1])
        except ValueError:
            return None

        words.remove(words[1])

    # Determine lead-time window.
    these_subwords = words[1].split('=')
    if these_subwords[0] != 'lead-time':
        return None
    if not these_subwords[1].endswith('sec'):
        return None

    lead_time_words = these_subwords[1].replace('sec', '').split('-')

    try:
        min_lead_time_sec = int(lead_time_words[0])
        max_lead_time_sec = int(lead_time_words[1])
    except ValueError:
        return None

    # Determine min/max linkage distance.
    these_subwords = words[2].split('=')
    if these_subwords[0] != 'distance':
        return None
    if not these_subwords[1].endswith('m'):
        return None

    distance_words = these_subwords[1].replace('m', '').split('-')

    try:
        min_link_distance_metres = float(int(distance_words[0]))
        max_link_distance_metres = float(int(distance_words[1]))
    except ValueError:
        return None

    target_param_dict = {
        MIN_LEAD_TIME_KEY: min_lead_time_sec,
        MAX_LEAD_TIME_KEY: max_lead_time_sec,
        MIN_LINKAGE_DISTANCE_KEY: min_link_distance_metres,
        MAX_LINKAGE_DISTANCE_KEY: max_link_distance_metres,
        MIN_FUJITA_RATING_KEY: min_fujita_rating,
        PERCENTILE_LEVEL_KEY: wind_speed_percentile_level,
        WIND_SPEED_CUTOFFS_KEY: None,
        EVENT_TYPE_KEY: event_type_string
    }

    wind_classifn = (
        event_type_string == linkage.WIND_EVENT_STRING and
        goal_string == CLASSIFICATION_STRING
    )

    if not wind_classifn:
        return target_param_dict

    these_subwords = words[3].split('=')
    if these_subwords[0] != 'cutoffs':
        return None
    if not these_subwords[1].endswith('kt'):
        return None

    cutoff_words = these_subwords[1].replace('kt', '').split('-')

    try:
        target_param_dict[WIND_SPEED_CUTOFFS_KEY] = numpy.array([
            int(c) for c in cutoff_words
        ], dtype=float)
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
    if target_param_dict[EVENT_TYPE_KEY] in [
            linkage.TORNADO_EVENT_STRING, linkage.TORNADOGENESIS_EVENT_STRING
    ]:
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
        percentile_level=DEFAULT_WIND_SPEED_PERCENTILE_LEVEL, test_mode=False):
    """For each storm object, creates regression target based on wind speed.

    :param storm_to_winds_table: See doc for `linkage.read_linkage_file`.
    :param min_lead_time_sec: See doc for `_check_target_params`.
    :param max_lead_time_sec: Same.
    :param min_link_distance_metres: Same.
    :param max_link_distance_metres: Same.
    :param percentile_level: Same.
    :param test_mode: Just leave this alone.
    :return: storm_to_winds_table: Same as input, but with additional column
        containing target values.  The name of this column is determined by
        `target_params_to_name`.
    """

    error_checking.assert_is_boolean(test_mode)
    if not test_mode:
        error_string = 'This method does not yet handle merging predecessors!'
        raise ValueError(error_string)

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

        these_good_indices = numpy.where(numpy.logical_and(
            these_good_time_flags, these_good_distance_flags
        ))[0]

        if len(these_good_indices == 0):
            # labels_m_s01[i] = 0.
            labels_m_s01[i] = INVALID_STORM_INTEGER
            continue

        these_wind_speeds_m_s01 = numpy.sqrt(
            storm_to_winds_table[linkage.U_WINDS_COLUMN].values[i][
                these_good_indices] ** 2 +
            storm_to_winds_table[linkage.V_WINDS_COLUMN].values[i][
                these_good_indices] ** 2
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

    return storm_to_winds_table.assign(**{
        target_name: labels_m_s01
    })


def create_wind_classification_targets(
        storm_to_winds_table, min_lead_time_sec=DEFAULT_MIN_LEAD_TIME_SEC,
        max_lead_time_sec=DEFAULT_MAX_LEAD_TIME_SEC,
        min_link_distance_metres=DEFAULT_MIN_LINK_DISTANCE_METRES,
        max_link_distance_metres=DEFAULT_MAX_LINK_DISTANCE_METRES,
        percentile_level=DEFAULT_WIND_SPEED_PERCENTILE_LEVEL,
        class_cutoffs_kt=DEFAULT_WIND_SPEED_CUTOFFS_KT, test_mode=False):
    """For each storm object, creates classification target based on wind speed.

    :param storm_to_winds_table: See doc for `linkage.read_linkage_file`.
    :param min_lead_time_sec: See doc for `_check_target_params`.
    :param max_lead_time_sec: Same.
    :param min_link_distance_metres: Same.
    :param max_link_distance_metres: Same.
    :param percentile_level: Same.
    :param class_cutoffs_kt: Same.
    :param test_mode: Just leave this alone.
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
        percentile_level=percentile_level, test_mode=test_mode)

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
        max_link_distance_metres=DEFAULT_MAX_LINK_DISTANCE_METRES,
        genesis_only=True, min_fujita_rating=0):
    """For each storm object, creates target based on tornado occurrence.

    :param storm_to_tornadoes_table: See doc for `linkage.read_linkage_file`.
    :param min_lead_time_sec: See doc for `_check_target_params`.
    :param max_lead_time_sec: Same.
    :param min_link_distance_metres: Same.
    :param max_link_distance_metres: Same.
    :param genesis_only: Same.
    :param min_fujita_rating: Same.
    :return: storm_to_tornadoes_table: Same as input, but with additional column
        containing target values.  The name of this column is determined by
        `target_params_to_name`.
    """

    num_storm_objects = len(storm_to_tornadoes_table.index)

    target_param_dict = _check_target_params(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres,
        tornadogenesis_only=genesis_only,
        min_fujita_rating=min_fujita_rating)

    min_link_distance_metres = target_param_dict[MIN_LINKAGE_DISTANCE_KEY]
    max_link_distance_metres = target_param_dict[MAX_LINKAGE_DISTANCE_KEY]

    end_of_period_indices = _find_storms_near_end_of_period(
        storm_to_events_table=storm_to_tornadoes_table,
        max_lead_time_sec=max_lead_time_sec)

    print((
        '{0:d} of {1:d} storm objects occur within {2:d} seconds of end '
        'of tracking period.'
    ).format(
        len(end_of_period_indices), num_storm_objects, max_lead_time_sec
    ))

    merging_predecessor_indices = numpy.where(
        storm_to_tornadoes_table[linkage.MERGING_PRED_FLAG_COLUMN].values
    )[0]

    print((
        '{0:d} of {1:d} storm objects are merging predecessors.'
    ).format(
        len(merging_predecessor_indices), num_storm_objects
    ))

    invalid_null_indices = numpy.concatenate((
        end_of_period_indices, merging_predecessor_indices
    ))
    tornado_classes = numpy.full(num_storm_objects, -1, dtype=int)

    for i in range(num_storm_objects):
        these_relative_times_sec = storm_to_tornadoes_table[
            linkage.RELATIVE_EVENT_TIMES_COLUMN
        ].values[i]

        these_link_distances_metres = storm_to_tornadoes_table[
            linkage.LINKAGE_DISTANCES_COLUMN
        ].values[i]

        these_fujita_ratings = numpy.array([
            tornado_io.fujita_string_to_int(f) for f in
            storm_to_tornadoes_table[linkage.FUJITA_RATINGS_COLUMN].values[i]
        ], dtype=int)

        these_good_time_flags = numpy.logical_and(
            these_relative_times_sec >= min_lead_time_sec,
            these_relative_times_sec <= max_lead_time_sec
        )
        these_good_distance_flags = numpy.logical_and(
            these_link_distances_metres >= min_link_distance_metres,
            these_link_distances_metres <= max_link_distance_metres
        )
        these_good_spacetime_flags = numpy.logical_and(
            these_good_time_flags, these_good_distance_flags
        )

        tornado_classes[i] = numpy.any(numpy.logical_and(
            these_good_spacetime_flags,
            these_fujita_ratings >= min_fujita_rating
        ))

        if i in invalid_null_indices and tornado_classes[i] == 0:
            tornado_classes[i] = INVALID_STORM_INTEGER

    target_name = target_params_to_name(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres,
        tornadogenesis_only=genesis_only,
        min_fujita_rating=min_fujita_rating)

    print((
        'Number of storm objects = {0:d} ... number with positive "{1:s}" label'
        ' = {2:d}'
    ).format(
        len(tornado_classes), target_name,
        int(numpy.round(numpy.sum(tornado_classes == 1)))
    ))

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

    if event_type_string == linkage.WIND_EVENT_STRING:
        pathless_file_name_prefix = 'wind_labels'
    elif event_type_string == linkage.TORNADOGENESIS_EVENT_STRING:
        pathless_file_name_prefix = 'tornadogenesis_labels'
    else:
        pathless_file_name_prefix = 'tornado_labels'

    if unix_time_sec is None:
        time_conversion.spc_date_string_to_unix_sec(spc_date_string)

        target_file_name = '{0:s}/{1:s}/{2:s}_{3:s}.nc'.format(
            top_directory_name, spc_date_string[:4], pathless_file_name_prefix,
            spc_date_string
        )
    else:
        spc_date_string = time_conversion.time_to_spc_date_string(unix_time_sec)
        valid_time_string = time_conversion.unix_sec_to_string(
            unix_time_sec, TIME_FORMAT)

        target_file_name = '{0:s}/{1:s}/{2:s}/{3:s}_{4:s}.nc'.format(
            top_directory_name, spc_date_string[:4], spc_date_string,
            pathless_file_name_prefix, valid_time_string
        )

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
        numpy.array(target_names), num_dimensions=1
    )

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

    full_id_strings = storm_to_events_table[
        tracking_utils.FULL_ID_COLUMN].values

    num_storm_objects = len(full_id_strings)
    num_id_characters = 0

    for i in range(num_storm_objects):
        num_id_characters = max([
            num_id_characters, len(full_id_strings[i])
        ])

    netcdf_dataset.createDimension(
        STORM_OBJECT_DIMENSION_KEY, num_storm_objects)
    netcdf_dataset.createDimension(CHARACTER_DIMENSION_KEY, num_id_characters)

    netcdf_dataset.createVariable(
        FULL_IDS_KEY, datatype='S1',
        dimensions=(STORM_OBJECT_DIMENSION_KEY, CHARACTER_DIMENSION_KEY)
    )

    string_type = 'S{0:d}'.format(num_id_characters)
    full_ids_char_array = netCDF4.stringtochar(numpy.array(
        full_id_strings, dtype=string_type
    ))
    netcdf_dataset.variables[FULL_IDS_KEY][:] = numpy.array(full_ids_char_array)

    netcdf_dataset.createVariable(
        VALID_TIMES_KEY, datatype=numpy.int32,
        dimensions=STORM_OBJECT_DIMENSION_KEY
    )
    netcdf_dataset.variables[VALID_TIMES_KEY][:] = storm_to_events_table[
        tracking_utils.VALID_TIME_COLUMN].values

    for this_target_name in target_names:
        netcdf_dataset.createVariable(
            this_target_name, datatype=numpy.float32,
            dimensions=STORM_OBJECT_DIMENSION_KEY
        )
        netcdf_dataset.variables[this_target_name][:] = storm_to_events_table[
            this_target_name].values

    netcdf_dataset.close()


def read_target_values(netcdf_file_name, target_names=None):
    """Reads target values from NetCDF file.

    E = number of examples (storm objects)
    T = number of target variables

    :param netcdf_file_name: Path to input file.
    :param target_names: 1-D list with names of target variables to read.  If
        None, will read all target variables.
    :return: storm_label_dict: Dictionary with the following keys.
    storm_label_dict['full_id_strings']: length-E list of full storm IDs.
    storm_label_dict['valid_times_unix_sec']: length-E numpy array of valid
        times.
    storm_label_dict['target_names']: length-T list with names of target
        variables.
    storm_label_dict['target_matrix']: E-by-T of target values (integer class
        labels).
    """

    netcdf_dataset = netcdf_io.open_netcdf(
        netcdf_file_name=netcdf_file_name, raise_error_if_fails=True)

    try:
        full_id_strings = netCDF4.chartostring(
            netcdf_dataset.variables[FULL_IDS_KEY][:]
        )
    except KeyError:
        full_id_strings = netCDF4.chartostring(
            netcdf_dataset.variables['storm_ids'][:]
        )

    valid_times_unix_sec = numpy.array(
        netcdf_dataset.variables[VALID_TIMES_KEY][:], dtype=int
    )

    if target_names is None:
        target_names = list(netcdf_dataset.variables.keys())
        target_names.remove(FULL_IDS_KEY)
        target_names.remove(VALID_TIMES_KEY)

    error_checking.assert_is_string_list(target_names)
    error_checking.assert_is_numpy_array(
        numpy.array(target_names), num_dimensions=1
    )

    num_storm_objects = len(full_id_strings)
    target_matrix = None

    for this_target_name in target_names:
        these_target_values = numpy.array(
            netcdf_dataset.variables[this_target_name][:], dtype=int
        )

        these_target_values = numpy.reshape(
            these_target_values, (num_storm_objects, 1)
        )

        if target_matrix is None:
            target_matrix = these_target_values + 0
        else:
            target_matrix = numpy.concatenate(
                (target_matrix, these_target_values), axis=1
            )

    netcdf_dataset.close()

    return {
        FULL_IDS_KEY: [str(f) for f in full_id_strings],
        VALID_TIMES_KEY: valid_times_unix_sec,
        TARGET_NAMES_KEY: target_names,
        TARGET_MATRIX_KEY: target_matrix
    }
