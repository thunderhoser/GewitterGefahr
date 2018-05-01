"""Deals with labels* for machine learning.

* Label = "dependent variable" = "target variable" = "outcome" = "predictand".
This is the variable that machine learning is trying to predict.  Examples are
max wind speed and tornado occurrence.
"""

import pickle
import os.path
import numpy
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import classification_utils as classifn_utils
from gewittergefahr.gg_utils import link_events_to_storms as events2storms
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
KT_TO_METRES_PER_SECOND = 1.852 / 3.6

REQUIRED_STORM_INPUT_COLUMNS = [
    tracking_utils.STORM_ID_COLUMN, tracking_utils.TIME_COLUMN,
    tracking_utils.TRACKING_END_TIME_COLUMN]
REQUIRED_STORM_OUTPUT_COLUMNS = [
    tracking_utils.STORM_ID_COLUMN, tracking_utils.TIME_COLUMN]

DEFAULT_MIN_LEAD_TIME_SEC = 0
DEFAULT_MAX_LEAD_TIME_SEC = 86400
DEFAULT_MIN_LINK_DISTANCE_METRES = 0.
DEFAULT_MAX_LINK_DISTANCE_METRES = 100000.
DEFAULT_WIND_SPEED_PERCENTILE_LEVEL = 100.
DEFAULT_WIND_SPEED_CLASS_CUTOFFS_KT = numpy.array([50.])

DISTANCE_PRECISION_METRES = 1.
PERCENTILE_LEVEL_PRECISION = 0.1
WIND_SPEED_CUTOFF_PRECISION_KT = 1.

MIN_LEAD_TIME_KEY = 'min_lead_time_sec'
MAX_LEAD_TIME_KEY = 'max_lead_time_sec'
MIN_LINKAGE_DISTANCE_KEY = 'min_link_distance_metres'
MAX_LINKAGE_DISTANCE_KEY = 'max_link_distance_metres'
EVENT_TYPE_KEY = 'event_type'
WIND_SPEED_PERCENTILE_LEVEL_KEY = 'wind_speed_percentile_level'
WIND_SPEED_CLASS_CUTOFFS_KEY = 'wind_speed_class_cutoffs_kt'

WIND_SPEED_PREFIX_FOR_REGRESSION_LABEL = 'wind-speed-m-s01'
WIND_SPEED_PREFIX_FOR_CLASSIFICATION_LABEL = 'wind-speed'
TORNADO_PREFIX_FOR_CLASSIFICATION_LABEL = 'tornado'
PREFIX_FOR_NUM_WIND_OBS_COLUMN = 'num-wind-observations'

REGRESSION_GOAL_STRING = 'regression'
CLASSIFICATION_GOAL_STRING = 'classification'
VALID_GOAL_STRINGS = [REGRESSION_GOAL_STRING, CLASSIFICATION_GOAL_STRING]


def _check_learning_goal(goal_string):
    """Ensures that learning goal is recognized.

    :param goal_string: Learning goal.
    :raises: ValueError: if `goal_string not in VALID_GOAL_STRINGS`.
    """

    error_checking.assert_is_string(goal_string)
    if goal_string not in VALID_GOAL_STRINGS:
        error_string = (
            '\n\n{0:s}Valid learning goals (listed above) do not include '
            '"{1:s}".').format(VALID_GOAL_STRINGS, goal_string)
        raise ValueError(error_string)


def _check_wind_speed_label_params(
        min_lead_time_sec, max_lead_time_sec, min_link_distance_metres,
        max_link_distance_metres, percentile_level):
    """Error-checks (and if necessary, rounds) parameters for wind-speed label.

    This label may be for either regression or classification.

    t_s = "storm time" = valid time of storm object
    t_w = "wind time" = time of wind observation
    "Lead time" = t_w - t_s

    :param min_lead_time_sec: Minimum lead time for wind observations.  Wind
        observations occurring before (t_s + min_lead_time_sec) will be ignored
        (not used to create the label).
    :param max_lead_time_sec: Max lead time for wind observations.  Wind
        observations occurring after (t_s + max_lead_time_sec) will be ignored.
    :param min_link_distance_metres: Minimum linkage distance for wind
        observations.  Wind observations closer to the storm boundary will be
        ignored.
    :param max_link_distance_metres: Max linkage distance for wind observations.
        Wind observations farther from the storm boundary will be ignored.
    :param percentile_level: The label for each storm object will be the
        [q]th-percentile wind speed for all observations in the given lead-time
        and linkage-distance ranges, where q = `wind_speed_percentile_level`.
    :return: parameter_dict: Dictionary with the following keys.
    parameter_dict['min_lead_time_sec']: Same as input.
    parameter_dict['max_lead_time_sec']: Same as input.
    parameter_dict['min_link_distance_metres']: Same as input, but rounded.
    parameter_dict['max_link_distance_metres']: Same as input, but rounded.
    parameter_dict['wind_speed_percentile_level']: Same as input, but rounded.
    """

    parameter_dict = check_label_params(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres)

    error_checking.assert_is_geq(percentile_level, 0.)
    error_checking.assert_is_leq(percentile_level, 100.)
    wind_speed_percentile_level = rounder.round_to_nearest(
        percentile_level, PERCENTILE_LEVEL_PRECISION)

    parameter_dict.update(
        {WIND_SPEED_PERCENTILE_LEVEL_KEY: wind_speed_percentile_level})
    return parameter_dict


def _check_cutoffs_for_wind_speed_classes(class_cutoffs_kt):
    """Error-checks (and if necessary, rounds) cutoffs for wind-speed classes.

    C = number of classes
    c = C - 1 = number of class cutoffs

    :param class_cutoffs_kt: length-c numpy array of class cutoffs in knots
        (nautical miles per hour).
    :return: class_cutoffs_kt: Same as input, except that values are unique and
        rounded.
    """

    class_cutoffs_kt = rounder.round_to_nearest(
        class_cutoffs_kt, WIND_SPEED_CUTOFF_PRECISION_KT)
    class_cutoffs_kt, _, _ = classifn_utils.classification_cutoffs_to_ranges(
        class_cutoffs_kt, non_negative_only=True)

    return class_cutoffs_kt


def _find_storms_near_end_of_tracking_period(
        storm_to_events_table, max_lead_time_sec):
    """Finds storm objects near end of tracking period.

    For storm objects within `max_lead_time_sec` of the end of the tracking
    period, we cannot confidently create labels.  For example, we cannot say
    "The storm produced no tornadoes within the lead-time range."

    :param storm_to_events_table: pandas DataFrame with at least the following
        columns.  Each row is one storm object.
    storm_to_events_table.tracking_end_time_unix_sec: End of tracking period for
        this storm object.
    storm_to_events_table.unix_time_sec: Valid time for this storm object.

    :param max_lead_time_sec: Max lead time for which labels are being created.
    :return: invalid_rows: 1-D numpy array with rows of invalid (too close to
        end of tracking period) storm objects.
    """

    times_before_end_of_tracking_sec = (
        storm_to_events_table[tracking_utils.TRACKING_END_TIME_COLUMN] -
        storm_to_events_table[tracking_utils.TIME_COLUMN])

    return numpy.where(
        times_before_end_of_tracking_sec < max_lead_time_sec)[0]


def check_label_params(
        min_lead_time_sec, max_lead_time_sec, min_link_distance_metres,
        max_link_distance_metres):
    """Error-checks (and if necessary, rounds) labeling parameters.

    t_s = "storm time" = valid time of storm object
    t_e = "event time" = time of wind observation or tornado
    "Lead time" = t_e - t_s

    :param min_lead_time_sec: Minimum lead time.  Events occurring before
        (t_s + min_lead_time_sec) will be ignored (not used to create the
        label).
    :param max_lead_time_sec: Max lead time.  Events occurring after
        (t_s + max_lead_time_sec) will be ignored.
    :param min_link_distance_metres: Minimum linkage distance.  Events closer to
        the storm boundary will be ignored.
    :param max_link_distance_metres: Max linkage distance.  Events farther from
        the storm boundary will be ignored.
    :return: parameter_dict: Dictionary with the following keys.
    parameter_dict['min_lead_time_sec']: Same as input.
    parameter_dict['max_lead_time_sec']: Same as input.
    parameter_dict['min_link_distance_metres']: Same as input, but rounded.
    parameter_dict['max_link_distance_metres']: Same as input, but rounded.
    """

    error_checking.assert_is_integer(min_lead_time_sec)
    error_checking.assert_is_geq(min_lead_time_sec, 0)
    error_checking.assert_is_integer(max_lead_time_sec)
    error_checking.assert_is_geq(max_lead_time_sec, min_lead_time_sec)
    error_checking.assert_is_geq(min_link_distance_metres, 0.)
    error_checking.assert_is_geq(max_link_distance_metres,
                                 min_link_distance_metres)

    min_link_distance_metres = rounder.round_to_nearest(
        min_link_distance_metres, DISTANCE_PRECISION_METRES)
    max_link_distance_metres = rounder.round_to_nearest(
        max_link_distance_metres, DISTANCE_PRECISION_METRES)

    return {
        MIN_LEAD_TIME_KEY: min_lead_time_sec,
        MAX_LEAD_TIME_KEY: max_lead_time_sec,
        MIN_LINKAGE_DISTANCE_KEY: min_link_distance_metres,
        MAX_LINKAGE_DISTANCE_KEY: max_link_distance_metres
    }


def get_column_name_for_regression_label(
        min_lead_time_sec, max_lead_time_sec, min_link_distance_metres,
        max_link_distance_metres, wind_speed_percentile_level):
    """Generates column name for regression label.

    :param min_lead_time_sec: See documentation for `_check_regression_params`.
    :param max_lead_time_sec: See doc for `_check_regression_params`.
    :param min_link_distance_metres: See doc for `_check_regression_params`.
    :param max_link_distance_metres: See doc for `_check_regression_params`.
    :param wind_speed_percentile_level: See doc for `_check_regression_params`.
    :return: column_name: Column name for regression label.
    """

    parameter_dict = _check_wind_speed_label_params(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres,
        percentile_level=wind_speed_percentile_level)

    min_lead_time_sec = parameter_dict[MIN_LEAD_TIME_KEY]
    max_lead_time_sec = parameter_dict[MAX_LEAD_TIME_KEY]
    min_link_distance_metres = parameter_dict[MIN_LINKAGE_DISTANCE_KEY]
    max_link_distance_metres = parameter_dict[MAX_LINKAGE_DISTANCE_KEY]
    wind_speed_percentile_level = parameter_dict[
        WIND_SPEED_PERCENTILE_LEVEL_KEY]

    return (
        '{0:s}_percentile={1:05.1f}_lead-time={2:04d}-{3:04d}sec_' +
        'distance={4:05d}-{5:05d}m').format(
            WIND_SPEED_PREFIX_FOR_REGRESSION_LABEL, wind_speed_percentile_level,
            min_lead_time_sec, max_lead_time_sec, int(min_link_distance_metres),
            int(max_link_distance_metres))


def get_column_name_for_num_wind_obs(
        min_lead_time_sec, max_lead_time_sec, min_link_distance_metres,
        max_link_distance_metres):
    """Generates column name for number of wind observations.

    Specifically, this is # of wind observations used to create regression or
    classification label.

    :param min_lead_time_sec: See documentation for `_check_regression_params`.
    :param max_lead_time_sec: See doc for `_check_regression_params`.
    :param min_link_distance_metres: See doc for `_check_regression_params`.
    :param max_link_distance_metres: See doc for `_check_regression_params`.
    :return: column_name: Column name for number of wind observations.
    """

    parameter_dict = check_label_params(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres)

    min_lead_time_sec = parameter_dict[MIN_LEAD_TIME_KEY]
    max_lead_time_sec = parameter_dict[MAX_LEAD_TIME_KEY]
    min_link_distance_metres = parameter_dict[MIN_LINKAGE_DISTANCE_KEY]
    max_link_distance_metres = parameter_dict[MAX_LINKAGE_DISTANCE_KEY]

    return (
        '{0:s}_lead-time={1:04d}-{2:04d}sec_distance={3:05d}-{4:05d}m').format(
            PREFIX_FOR_NUM_WIND_OBS_COLUMN, min_lead_time_sec,
            max_lead_time_sec, int(min_link_distance_metres),
            int(max_link_distance_metres))


def get_column_name_for_classification_label(
        min_lead_time_sec, max_lead_time_sec, min_link_distance_metres,
        max_link_distance_metres, event_type_string,
        wind_speed_percentile_level=None, wind_speed_class_cutoffs_kt=None):
    """Generates column name for classification label.

    :param min_lead_time_sec: See documentation for `_check_regression_params`.
    :param max_lead_time_sec: See doc for `_check_regression_params`.
    :param min_link_distance_metres: See doc for `_check_regression_params`.
    :param max_link_distance_metres: See doc for `_check_regression_params`.
    :param event_type_string: Event type ("wind" or "tornado").
    :param wind_speed_percentile_level: See doc for `_check_regression_params`.
    :param wind_speed_class_cutoffs_kt: See doc for
        `_check_cutoffs_for_wind_speed_classes`.
    :return: column_name: Column name for classification label.
    """

    events2storms.check_event_type(event_type_string)

    if event_type_string == events2storms.WIND_EVENT_TYPE_STRING:
        parameter_dict = _check_wind_speed_label_params(
            min_lead_time_sec=min_lead_time_sec,
            max_lead_time_sec=max_lead_time_sec,
            min_link_distance_metres=min_link_distance_metres,
            max_link_distance_metres=max_link_distance_metres,
            percentile_level=wind_speed_percentile_level)
    else:
        parameter_dict = check_label_params(
            min_lead_time_sec=min_lead_time_sec,
            max_lead_time_sec=max_lead_time_sec,
            min_link_distance_metres=min_link_distance_metres,
            max_link_distance_metres=max_link_distance_metres)

    min_lead_time_sec = parameter_dict[MIN_LEAD_TIME_KEY]
    max_lead_time_sec = parameter_dict[MAX_LEAD_TIME_KEY]
    min_link_distance_metres = parameter_dict[MIN_LINKAGE_DISTANCE_KEY]
    max_link_distance_metres = parameter_dict[MAX_LINKAGE_DISTANCE_KEY]

    if event_type_string == events2storms.WIND_EVENT_TYPE_STRING:
        wind_speed_percentile_level = parameter_dict[
            WIND_SPEED_PERCENTILE_LEVEL_KEY]

        wind_speed_class_cutoffs_kt = _check_cutoffs_for_wind_speed_classes(
            wind_speed_class_cutoffs_kt)
        num_cutoffs = len(wind_speed_class_cutoffs_kt)

        column_name = (
            '{0:s}_percentile={1:05.1f}_lead-time={2:04d}-{3:04d}sec_' +
            'distance={4:05d}-{5:05d}m_cutoffs=').format(
                WIND_SPEED_PREFIX_FOR_CLASSIFICATION_LABEL,
                wind_speed_percentile_level, min_lead_time_sec,
                max_lead_time_sec, int(min_link_distance_metres),
                int(max_link_distance_metres))

        for k in range(num_cutoffs):
            if k == 0:
                column_name += '{0:02d}'.format(
                    int(wind_speed_class_cutoffs_kt[k]))
            else:
                column_name += '-{0:02d}'.format(
                    int(wind_speed_class_cutoffs_kt[k]))

        column_name += 'kt'

    else:
        column_name = (
            '{0:s}_lead-time={1:04d}-{2:04d}sec_distance='
            '{3:05d}-{4:05d}m').format(
                TORNADO_PREFIX_FOR_CLASSIFICATION_LABEL, min_lead_time_sec,
                max_lead_time_sec, int(min_link_distance_metres),
                int(max_link_distance_metres))

    return column_name


def column_name_to_label_params(column_name):
    """Parses labeling parameters from column name.

    :param column_name: Column name (generated by either
        `get_column_name_for_regression_label` or
        `get_column_name_for_classification_label`).
    :return: parameter_dict: Dictionary with the following keys.
    parameter_dict['min_lead_time_sec']: See documentation for
        `check_label_params`.
    parameter_dict['max_lead_time_sec']: See doc for `check_label_params`.
    parameter_dict['min_link_distance_metres']: See doc for
        `check_label_params`.
    parameter_dict['max_link_distance_metres']: See doc for
        `check_label_params`.
    parameter_dict['event_type']: Either "wind" or "tornado".
    parameter_dict['wind_speed_percentile_level']: See doc for
        `_check_wind_speed_label_params`.  If event type is not wind, this is
        None.
    parameter_dict['wind_speed_class_cutoffs_kt']: 1-D numpy array of class
        cutoffs in knots (nautical miles per hour).  If learning goal is not
        wind-speed classification, this is None.
    """

    column_name_parts = column_name.split('_')

    # Determine event type and goal (may be wind speed and regression, wind
    # speed and classification, or tornado and classification).
    if column_name_parts[0] == WIND_SPEED_PREFIX_FOR_REGRESSION_LABEL:
        goal_string = REGRESSION_GOAL_STRING
        event_type_string = events2storms.WIND_EVENT_TYPE_STRING
        if len(column_name_parts) != 4:
            return None

    elif column_name_parts[0] == WIND_SPEED_PREFIX_FOR_CLASSIFICATION_LABEL:
        goal_string = CLASSIFICATION_GOAL_STRING
        event_type_string = events2storms.WIND_EVENT_TYPE_STRING
        if len(column_name_parts) != 5:
            return None

    elif column_name_parts[0] == TORNADO_PREFIX_FOR_CLASSIFICATION_LABEL:
        goal_string = CLASSIFICATION_GOAL_STRING
        event_type_string = events2storms.TORNADO_EVENT_TYPE_STRING
        if len(column_name_parts) != 3:
            return None

    else:
        return None

    # Determine wind-speed percentile.
    if event_type_string == events2storms.WIND_EVENT_TYPE_STRING:
        percentile_part = column_name_parts[1]
        if not percentile_part.startswith('percentile='):
            return None

        percentile_part = percentile_part.replace('percentile=', '')
        try:
            wind_speed_percentile_level = float(percentile_part)
        except ValueError:
            return None

        column_name_parts.remove(column_name_parts[1])
    else:
        wind_speed_percentile_level = None

    # Determine min/max lead times.
    lead_time_part = column_name_parts[1]
    if not lead_time_part.startswith('lead-time='):
        return None
    if not lead_time_part.endswith('sec'):
        return None

    lead_time_part = lead_time_part.replace('lead-time=', '').replace('sec', '')
    lead_time_parts = lead_time_part.split('-')
    if len(lead_time_parts) != 2:
        return None

    try:
        min_lead_time_sec = int(lead_time_parts[0])
        max_lead_time_sec = int(lead_time_parts[1])
    except ValueError:
        return None

    # Determine min/max linkage distances.
    distance_part = column_name_parts[2]
    if not distance_part.startswith('distance='):
        return None
    if not distance_part.endswith('m'):
        return None

    distance_part = distance_part.replace('distance=', '').replace('m', '')
    distance_parts = distance_part.split('-')
    if len(distance_parts) != 2:
        return None

    try:
        min_link_distance_metres = float(int(distance_parts[0]))
        max_link_distance_metres = float(int(distance_parts[1]))
    except ValueError:
        return None

    # Determine cutoffs for wind-speed classes.
    if (event_type_string == events2storms.WIND_EVENT_TYPE_STRING
            and goal_string == CLASSIFICATION_GOAL_STRING):

        class_cutoff_part = column_name_parts[3]
        if not class_cutoff_part.startswith('cutoffs='):
            return None
        if not class_cutoff_part.endswith('kt'):
            return None

        class_cutoff_part = class_cutoff_part.replace('cutoffs=', '').replace(
            'kt', '')
        class_cutoff_parts = class_cutoff_part.split('-')

        try:
            wind_speed_class_cutoffs_kt = numpy.array(
                [int(c) for c in class_cutoff_parts]).astype(float)
        except ValueError:
            return None

    else:
        wind_speed_class_cutoffs_kt = None

    return {
        MIN_LEAD_TIME_KEY: min_lead_time_sec,
        MAX_LEAD_TIME_KEY: max_lead_time_sec,
        MIN_LINKAGE_DISTANCE_KEY: min_link_distance_metres,
        MAX_LINKAGE_DISTANCE_KEY: max_link_distance_metres,
        EVENT_TYPE_KEY: event_type_string,
        WIND_SPEED_PERCENTILE_LEVEL_KEY: wind_speed_percentile_level,
        WIND_SPEED_CLASS_CUTOFFS_KEY: wind_speed_class_cutoffs_kt
    }


def get_columns_with_labels(label_table, goal_string, event_type_string):
    """Returns names of columns with labels of the given type.

    :param label_table: pandas DataFrame.
    :param goal_string: Learning goal (either "regression" or "classification").
    :param event_type_string: Event type (either "wind" or "tornado").
    :return: label_column_names: 1-D list containing names of columns with
        labels of the given type.  If there are no labels of the given type,
        this is None.
    """

    _check_learning_goal(goal_string)
    events2storms.check_event_type(event_type_string)

    column_names = list(label_table)
    label_column_names = None

    for this_column_name in column_names:
        this_parameter_dict = column_name_to_label_params(this_column_name)
        if this_parameter_dict is None:
            continue

        if this_parameter_dict[EVENT_TYPE_KEY] != event_type_string:
            continue

        if goal_string == REGRESSION_GOAL_STRING:
            if this_parameter_dict[WIND_SPEED_CLASS_CUTOFFS_KEY] is not None:
                continue
        else:
            if (event_type_string == events2storms.WIND_EVENT_TYPE_STRING and
                    this_parameter_dict[WIND_SPEED_CLASS_CUTOFFS_KEY] is None):
                continue

        if label_column_names is None:
            label_column_names = [this_column_name]
        else:
            label_column_names.append(this_column_name)

    return label_column_names


def get_columns_with_num_wind_obs(label_table, label_column_names):
    """Returns names of columns with # wind observations used to create label.

    :param label_table: pandas DataFrame.
    :param label_column_names: 1-D list with names of columns containing
        wind-speed-based labels.  This method will find the corresponding
        "number of wind observations" columns.
    :return: num_wind_obs_column_names: 1-D list with names of columns
        containing number of wind observations used to create label.
    :raises: ValueError: if `label_table` is missing any "number of wind
        observations" column corresponding to a column in `label_column_names`.
    """

    error_checking.assert_is_string_list(label_column_names)
    error_checking.assert_is_numpy_array(
        numpy.array(label_column_names), num_dimensions=1)

    num_wind_obs_column_names = set()

    for this_label_column_name in label_column_names:
        this_parameter_dict = column_name_to_label_params(
            this_label_column_name)

        if (this_parameter_dict[EVENT_TYPE_KEY] !=
                events2storms.WIND_EVENT_TYPE_STRING):
            continue

        this_num_wind_obs_column_name = get_column_name_for_num_wind_obs(
            min_lead_time_sec=this_parameter_dict[MIN_LEAD_TIME_KEY],
            max_lead_time_sec=this_parameter_dict[MAX_LEAD_TIME_KEY],
            min_link_distance_metres=this_parameter_dict[
                MIN_LINKAGE_DISTANCE_KEY],
            max_link_distance_metres=this_parameter_dict[
                MAX_LINKAGE_DISTANCE_KEY])

        if this_num_wind_obs_column_name not in list(label_table):
            error_string = (
                'label_table has column "{0:s}" but does not have column '
                '"{1:s}".').format(this_label_column_name,
                                   this_num_wind_obs_column_name)
            raise ValueError(error_string)

        num_wind_obs_column_names.add(this_num_wind_obs_column_name)

    num_wind_obs_column_names = list(num_wind_obs_column_names)
    if not len(num_wind_obs_column_names):
        num_wind_obs_column_names = None

    return num_wind_obs_column_names


def check_wind_speed_label_table(wind_speed_label_table):
    """Ensures that pandas DataFrame contains expected columns.

    :param wind_speed_label_table: pandas DataFrame, where each row contains
        wind-speed-based labels for one storm object.
    :return: label_column_names: 1-D list with names of columns containing
        wind-speed-based labels.
    :return: num_wind_obs_column_names: 1-D list with names of columns
        containing number of wind observations used to create label.
    :raises: TypeError: if `wind_speed_label_table` contains no columns with
        wind-speed-based labels.
    :raises: ValueError: if `wind_speed_label_table` contains a classification
        label but not the corresponding regression label.
    """

    error_checking.assert_columns_in_dataframe(
        wind_speed_label_table, REQUIRED_STORM_OUTPUT_COLUMNS)

    classification_column_names = get_columns_with_labels(
        label_table=wind_speed_label_table,
        goal_string=CLASSIFICATION_GOAL_STRING,
        event_type_string=events2storms.WIND_EVENT_TYPE_STRING)
    if classification_column_names is None:
        classification_column_names = []

    for this_classifn_column_name in classification_column_names:
        this_parameter_dict = column_name_to_label_params(
            this_classifn_column_name)

        this_regression_column_name = (
            get_column_name_for_regression_label(
                min_lead_time_sec=this_parameter_dict[MIN_LEAD_TIME_KEY],
                max_lead_time_sec=this_parameter_dict[MAX_LEAD_TIME_KEY],
                min_link_distance_metres=this_parameter_dict[
                    MIN_LINKAGE_DISTANCE_KEY],
                max_link_distance_metres=this_parameter_dict[
                    MAX_LINKAGE_DISTANCE_KEY],
                wind_speed_percentile_level=this_parameter_dict[
                    WIND_SPEED_PERCENTILE_LEVEL_KEY]))

        if this_regression_column_name in list(wind_speed_label_table):
            continue

        error_string = (
            'wind_speed_label_table contains classification label ("{0:s}") '
            'but not the corresponding regression label ("{1:s}").').format(
                this_classifn_column_name, this_regression_column_name)
        raise ValueError(error_string)

    regression_column_names = get_columns_with_labels(
        label_table=wind_speed_label_table, goal_string=REGRESSION_GOAL_STRING,
        event_type_string=events2storms.WIND_EVENT_TYPE_STRING)
    if regression_column_names is None:
        regression_column_names = []

    label_column_names = classification_column_names + regression_column_names
    if not len(label_column_names):
        raise TypeError('wind_speed_label_table contains no columns with '
                        'wind-speed-based label.')

    num_wind_obs_column_names = get_columns_with_num_wind_obs(
        label_table=wind_speed_label_table,
        label_column_names=label_column_names)

    return label_column_names, num_wind_obs_column_names


def check_tornado_label_table(tornado_label_table):
    """Ensures that pandas DataFrame contains expected columns.

    :param tornado_label_table: pandas DataFrame, where each row contains
        tornado-based labels for one storm object.
    :return: label_column_names: 1-D list with names of columns containing
        tornado-based labels.
    :raises: TypeError: if `tornado_label_table` contains no columns with
        tornado-based labels.
    """

    error_checking.assert_columns_in_dataframe(
        tornado_label_table, REQUIRED_STORM_OUTPUT_COLUMNS)

    label_column_names = get_columns_with_labels(
        label_table=tornado_label_table, goal_string=CLASSIFICATION_GOAL_STRING,
        event_type_string=events2storms.TORNADO_EVENT_TYPE_STRING)

    if label_column_names is None:
        raise TypeError(
            'tornado_label_table contains no columns with tornado-based label.')

    return label_column_names


def label_wind_speed_for_regression(
        storm_to_winds_table, min_lead_time_sec=DEFAULT_MIN_LEAD_TIME_SEC,
        max_lead_time_sec=DEFAULT_MAX_LEAD_TIME_SEC,
        min_link_distance_metres=DEFAULT_MIN_LINK_DISTANCE_METRES,
        max_link_distance_metres=DEFAULT_MAX_LINK_DISTANCE_METRES,
        percentile_level=DEFAULT_WIND_SPEED_PERCENTILE_LEVEL):
    """For each storm object, creates regression label based on wind speed.

    :param storm_to_winds_table: pandas DataFrame with columns listed in
        `link_events_to_storms.write_storm_to_winds_table`.
    :param min_lead_time_sec: See documentation for
        `_check_wind_speed_label_params`.
    :param max_lead_time_sec: See doc for `_check_wind_speed_label_params`.
    :param min_link_distance_metres: See doc for
        `_check_wind_speed_label_params`.
    :param max_link_distance_metres: See doc for
        `_check_wind_speed_label_params`.
    :param percentile_level: See doc for `_check_wind_speed_label_params`.
    :return: storm_to_winds_table: Same as input, with the following exceptions.
        [1] Contains new column with regression labels (column name generated by
            `get_column_name_for_regression_label`).
        [2] Contains new column with number of wind observations used to label
            each storm object (column name generated by
            `get_column_name_for_num_wind_obs`).
    """

    parameter_dict = _check_wind_speed_label_params(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres,
        percentile_level=percentile_level)

    min_lead_time_sec = parameter_dict[MIN_LEAD_TIME_KEY]
    max_lead_time_sec = parameter_dict[MAX_LEAD_TIME_KEY]
    min_link_distance_metres = parameter_dict[MIN_LINKAGE_DISTANCE_KEY]
    max_link_distance_metres = parameter_dict[MAX_LINKAGE_DISTANCE_KEY]
    percentile_level = parameter_dict[WIND_SPEED_PERCENTILE_LEVEL_KEY]

    invalid_storm_indices = _find_storms_near_end_of_tracking_period(
        storm_to_winds_table, max_lead_time_sec)

    num_storm_objects = len(storm_to_winds_table.index)
    labels_m_s01 = numpy.full(num_storm_objects, numpy.nan)
    numbers_of_wind_obs = numpy.full(num_storm_objects, -1, dtype=int)

    for i in range(num_storm_objects):
        if i in invalid_storm_indices:
            continue

        these_relative_wind_times_sec = storm_to_winds_table[
            events2storms.RELATIVE_EVENT_TIMES_COLUMN].values[i]
        these_link_distances_metres = storm_to_winds_table[
            events2storms.LINKAGE_DISTANCES_COLUMN].values[i]

        these_valid_time_flags = numpy.logical_and(
            these_relative_wind_times_sec >= min_lead_time_sec,
            these_relative_wind_times_sec <= max_lead_time_sec)
        these_valid_distance_flags = numpy.logical_and(
            these_link_distances_metres >= min_link_distance_metres,
            these_link_distances_metres <= max_link_distance_metres)
        these_valid_wind_flags = numpy.logical_and(
            these_valid_time_flags, these_valid_distance_flags)

        if not numpy.any(these_valid_wind_flags):
            labels_m_s01[i] = 0.
            numbers_of_wind_obs[i] = 0
            continue

        these_valid_wind_indices = numpy.where(these_valid_wind_flags)[0]
        numbers_of_wind_obs[i] = len(these_valid_wind_indices)

        these_u_winds_m_s01 = storm_to_winds_table[
            events2storms.U_WINDS_COLUMN].values[i][these_valid_wind_indices]
        these_v_winds_m_s01 = storm_to_winds_table[
            events2storms.V_WINDS_COLUMN].values[i][these_valid_wind_indices]

        these_wind_speeds_m_s01 = numpy.sqrt(
            these_u_winds_m_s01 ** 2 + these_v_winds_m_s01 ** 2)
        labels_m_s01[i] = numpy.percentile(
            these_wind_speeds_m_s01, percentile_level)

    label_column_name = get_column_name_for_regression_label(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres,
        wind_speed_percentile_level=percentile_level)

    num_wind_obs_column_name = get_column_name_for_num_wind_obs(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres)

    argument_dict = {label_column_name: labels_m_s01,
                     num_wind_obs_column_name: numbers_of_wind_obs}
    return storm_to_winds_table.assign(**argument_dict)


def label_wind_speed_for_classification(
        storm_to_winds_table, min_lead_time_sec=DEFAULT_MIN_LEAD_TIME_SEC,
        max_lead_time_sec=DEFAULT_MAX_LEAD_TIME_SEC,
        min_link_distance_metres=DEFAULT_MIN_LINK_DISTANCE_METRES,
        max_link_distance_metres=DEFAULT_MAX_LINK_DISTANCE_METRES,
        percentile_level=DEFAULT_WIND_SPEED_PERCENTILE_LEVEL,
        class_cutoffs_kt=DEFAULT_WIND_SPEED_CLASS_CUTOFFS_KT):
    """For each storm object, creates classification label based on wind speed.

    :param storm_to_winds_table: pandas DataFrame with columns listed in
        `link_events_to_storms.write_storm_to_winds_table`.
    :param min_lead_time_sec: See documentation for
        `_check_wind_speed_label_params`.
    :param max_lead_time_sec: See doc for `_check_wind_speed_label_params`.
    :param min_link_distance_metres: See doc for
        `_check_wind_speed_label_params`.
    :param max_link_distance_metres: See doc for
        `_check_wind_speed_label_params`.
    :param percentile_level: See doc for `_check_wind_speed_label_params`.
    :param class_cutoffs_kt: See doc for
        `_check_cutoffs_for_wind_speed_classes`.
    :return: storm_to_winds_table: Same as input, with the following exceptions.
        [1] Contains new column with regression labels (column name generated by
            `get_column_name_for_regression_label`).
        [2] Contains new column with number of wind observations used to label
            each storm object (column name generated by
            `get_column_name_for_num_wind_obs`).
        [3] Contains new column "num_wind_observations", indicating how many
            observations were factored into each label.
    """

    class_cutoffs_m_s01 = _check_cutoffs_for_wind_speed_classes(
        class_cutoffs_kt) * KT_TO_METRES_PER_SECOND

    storm_to_winds_table = label_wind_speed_for_regression(
        storm_to_winds_table=storm_to_winds_table,
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres,
        percentile_level=percentile_level)

    regression_label_column_name = get_column_name_for_regression_label(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres,
        wind_speed_percentile_level=percentile_level)

    regression_labels_m_s01 = storm_to_winds_table[
        regression_label_column_name].values
    invalid_storm_indices = numpy.where(numpy.isnan(regression_labels_m_s01))[0]
    regression_labels_m_s01[invalid_storm_indices] = 0.

    storm_classes = classifn_utils.classify_values(
        regression_labels_m_s01, class_cutoffs=class_cutoffs_m_s01,
        non_negative_only=True)
    storm_classes[invalid_storm_indices] = -1

    classification_label_column_name = get_column_name_for_classification_label(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres,
        event_type_string=events2storms.WIND_EVENT_TYPE_STRING,
        wind_speed_percentile_level=percentile_level,
        wind_speed_class_cutoffs_kt=class_cutoffs_kt)

    return storm_to_winds_table.assign(
        **{classification_label_column_name: storm_classes})


def label_tornado_occurrence(
        storm_to_tornadoes_table, min_lead_time_sec=DEFAULT_MIN_LEAD_TIME_SEC,
        max_lead_time_sec=DEFAULT_MAX_LEAD_TIME_SEC,
        min_link_distance_metres=DEFAULT_MIN_LINK_DISTANCE_METRES,
        max_link_distance_metres=DEFAULT_MAX_LINK_DISTANCE_METRES):
    """For each storm object, creates binary label based on tornado occurrence.

    :param storm_to_tornadoes_table: pandas DataFrame with columns listed in
        `link_events_to_storms.write_storm_to_tornadoes_table`.
    :param min_lead_time_sec: See documentation for
        `_check_wind_speed_label_params`.
    :param max_lead_time_sec: See doc for `check_label_params`.
    :param min_link_distance_metres: See doc for `check_label_params`.
    :param max_link_distance_metres: See doc for `check_label_params`.
    :return: storm_to_tornadoes_table: Same as input, with the following
        exceptions.
        [1] May have fewer rows (storm objects near the end of the tracking
            period are removed).
        [2] Contains new column with tornado labels (column name generated by
            `get_column_name_for_classification_label`).
    """

    parameter_dict = check_label_params(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres)

    min_lead_time_sec = parameter_dict[MIN_LEAD_TIME_KEY]
    max_lead_time_sec = parameter_dict[MAX_LEAD_TIME_KEY]
    min_link_distance_metres = parameter_dict[MIN_LINKAGE_DISTANCE_KEY]
    max_link_distance_metres = parameter_dict[MAX_LINKAGE_DISTANCE_KEY]

    invalid_storm_indices = _find_storms_near_end_of_tracking_period(
        storm_to_tornadoes_table, max_lead_time_sec)

    num_storm_objects = len(storm_to_tornadoes_table.index)
    tornado_classes = numpy.full(num_storm_objects, -1, dtype=int)

    for i in range(num_storm_objects):
        if i in invalid_storm_indices:
            continue

        these_relative_tornado_times_sec = storm_to_tornadoes_table[
            events2storms.RELATIVE_EVENT_TIMES_COLUMN].values[i]
        these_link_distance_metres = storm_to_tornadoes_table[
            events2storms.LINKAGE_DISTANCES_COLUMN].values[i]

        these_valid_time_flags = numpy.logical_and(
            these_relative_tornado_times_sec >= min_lead_time_sec,
            these_relative_tornado_times_sec <= max_lead_time_sec)
        these_valid_distance_flags = numpy.logical_and(
            these_link_distance_metres >= min_link_distance_metres,
            these_link_distance_metres <= max_link_distance_metres)
        these_valid_tornado_flags = numpy.logical_and(
            these_valid_time_flags, these_valid_distance_flags)

        tornado_classes[i] = numpy.any(these_valid_tornado_flags)

    label_column_name = get_column_name_for_classification_label(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres,
        event_type_string=events2storms.TORNADO_EVENT_TYPE_STRING)

    return storm_to_tornadoes_table.assign(
        **{label_column_name: tornado_classes})


def find_label_file(
        top_directory_name, event_type_string, raise_error_if_missing=True,
        unix_time_sec=None, spc_date_string=None):
    """Finds label file for either one time step or one SPC date.

    In other words, the file should be one written by `write_wind_speed_labels`
    or `write_tornado_labels`.

    :param top_directory_name: Name of top-level directory with label files.
    :param event_type_string: Either "wind" or "tornado".
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        raise_error_if_missing = True, this method will error out.
    :param unix_time_sec: Valid time.
    :param spc_date_string: [used only if unix_time_sec is None]
        SPC date (format "yyyymmdd").
    :return: label_file_name: File path.  If file is missing and
        raise_error_if_missing = False, this is the *expected* path.
    :raises: ValueError: if file is missing and raise_error_if_missing = True.
    """

    error_checking.assert_is_string(top_directory_name)
    events2storms.check_event_type(event_type_string)
    error_checking.assert_is_boolean(raise_error_if_missing)

    if unix_time_sec is None:
        time_conversion.spc_date_string_to_unix_sec(spc_date_string)

        if event_type_string == events2storms.WIND_EVENT_TYPE_STRING:
            label_file_name = '{0:s}/{1:s}/wind_labels_{2:s}.p'.format(
                top_directory_name, spc_date_string[:4], spc_date_string)
        else:
            label_file_name = '{0:s}/{1:s}/tornado_labels_{2:s}.p'.format(
                top_directory_name, spc_date_string[:4], spc_date_string)
    else:
        spc_date_string = time_conversion.time_to_spc_date_string(unix_time_sec)

        if event_type_string == events2storms.WIND_EVENT_TYPE_STRING:
            label_file_name = '{0:s}/{1:s}/{2:s}/wind_labels_{3:s}.p'.format(
                top_directory_name, spc_date_string[:4], spc_date_string,
                time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT))
        else:
            label_file_name = '{0:s}/{1:s}/{2:s}/tornado_labels_{3:s}.p'.format(
                top_directory_name, spc_date_string[:4], spc_date_string,
                time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT))

    if raise_error_if_missing and not os.path.isfile(label_file_name):
        error_string = 'Cannot find label file.  Expected at: "{0:s}"'.format(
            label_file_name)
        raise ValueError(error_string)

    return label_file_name


def write_wind_speed_labels(storm_to_winds_table, pickle_file_name):
    """Writes wind-speed-based labels to a Pickle file.

    :param storm_to_winds_table: pandas DataFrame created by
        `label_wind_speed_for_regression` or
        `label_wind_speed_for_classification`.
    :param pickle_file_name: Path to output file.
    """

    label_column_names, num_wind_obs_column_names = (
        check_wind_speed_label_table(storm_to_winds_table))
    columns_to_write = (
        label_column_names + num_wind_obs_column_names +
        events2storms.get_columns_to_write(
            storm_to_winds_table=storm_to_winds_table))

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(storm_to_winds_table[columns_to_write], pickle_file_handle)
    pickle_file_handle.close()


def write_tornado_labels(storm_to_tornadoes_table, pickle_file_name):
    """Writes tornado-based labels to a Pickle file.

    :param storm_to_tornadoes_table: pandas DataFrame created by
        `label_tornado_occurrence`.
    :param pickle_file_name: Path to output file.
    """

    label_column_names = check_tornado_label_table(storm_to_tornadoes_table)
    columns_to_write = label_column_names + events2storms.get_columns_to_write(
        storm_to_tornadoes_table=storm_to_tornadoes_table)

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(storm_to_tornadoes_table[columns_to_write], pickle_file_handle)
    pickle_file_handle.close()


def read_wind_speed_labels(pickle_file_name):
    """Reads wind-speed-based labels from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: storm_to_winds_table: pandas DataFrame created by
        `label_wind_speed_for_regression` or
        `label_wind_speed_for_classification`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    storm_to_winds_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    check_wind_speed_label_table(storm_to_winds_table)
    return storm_to_winds_table


def read_tornado_labels(pickle_file_name):
    """Reads tornado-based labels from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: storm_to_tornadoes_table: pandas DataFrame created by
        `label_tornado_occurrence`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    storm_to_tornadoes_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    check_tornado_label_table(storm_to_tornadoes_table)
    return storm_to_tornadoes_table
