"""Labels each storm object for machine learning.

--- DEFINITIONS ---

Label = "dependent variable" = "target variable" = "outcome" = "predictand".
This is the variable that machine learning is trying to predict.  An example is
the max wind speed associated with the storm object.
"""

import pickle
import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.linkage import storm_to_winds
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import classification_utils as classifn_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

KT_TO_METRES_PER_SECOND = 1.852 / 3.6

NUM_OBSERVATIONS_FOR_LABEL_COLUMN = 'num_observations_for_label'
MANDATORY_COLUMNS = [
    tracking_io.STORM_ID_COLUMN, tracking_io.TIME_COLUMN,
    tracking_io.TRACKING_END_TIME_COLUMN, NUM_OBSERVATIONS_FOR_LABEL_COLUMN]

DEFAULT_MIN_LEAD_TIME_SEC = 0
DEFAULT_MAX_LEAD_TIME_SEC = 86400  # Greater than will ever occur.
DEFAULT_MIN_DISTANCE_METRES = 0.
DEFAULT_MAX_DISTANCE_METRES = 100000.  # Greater than will ever occur.
DEFAULT_PERCENTILE_LEVEL = 100.
DEFAULT_CLASS_CUTOFFS_KT = numpy.array([50.])

DISTANCE_PRECISION_METRES = 1.
PERCENTILE_LEVEL_PRECISION = 0.1
CLASS_CUTOFF_PRECISION_KT = 1.
MAX_CLASS_MAX_KT = 1000.

MIN_LEAD_TIME_NAME = 'min_lead_time_sec'
MAX_LEAD_TIME_NAME = 'max_lead_time_sec'
MIN_DISTANCE_NAME = 'min_distance_metres'
MAX_DISTANCE_NAME = 'max_distance_metres'
PERCENTILE_LEVEL_NAME = 'percentile_level'
CLASS_CUTOFFS_NAME = 'class_cutoffs_kt'

PREFIX_FOR_REGRESSION_LABEL = 'wind_speed_m_s01'
PREFIX_FOR_CLASSIFICATION_LABEL = 'wind_speed'


def _check_regression_params(
        min_lead_time_sec=None, max_lead_time_sec=None,
        min_distance_metres=None, max_distance_metres=None,
        percentile_level=None):
    """Error-checks (and if necessary, rounds) parameters for regression labels.

    t = time of a given storm object

    :param min_lead_time_sec: Minimum lead time (wind time minus storm-object
        time).  Wind observations occurring before t + min_lead_time_sec are
        ignored.
    :param max_lead_time_sec: Maximum lead time (wind time minus storm-object
        time).  Wind observations occurring after t + max_lead_time_sec are
        ignored.
    :param min_distance_metres: Minimum distance between storm boundary and wind
        observations.  Wind observations nearer to the storm are ignored.
    :param max_distance_metres: Maximum distance between storm boundary and wind
        observations.  Wind observations farther from the storm are ignored.
    :param percentile_level: The label for each storm object will be the [q]th-
        percentile speed of wind observations in the given time and distance
        ranges, where q = percentile_level.
    :return: parameter_dict: Dictionary with the following keys.
    parameter_dict['min_lead_time_sec']: Same as input, but maybe rounded.
    parameter_dict['max_lead_time_sec']: Same as input, but maybe rounded.
    parameter_dict['min_distance_metres']: Same as input, but maybe rounded.
    parameter_dict['max_distance_metres']: Same as input, but maybe rounded.
    parameter_dict['percentile_level']: Same as input, but maybe rounded.
    """

    error_checking.assert_is_integer(min_lead_time_sec)
    error_checking.assert_is_geq(min_lead_time_sec, 0)
    error_checking.assert_is_integer(max_lead_time_sec)
    error_checking.assert_is_geq(max_lead_time_sec, min_lead_time_sec)
    error_checking.assert_is_geq(min_distance_metres, 0.)
    error_checking.assert_is_geq(max_distance_metres, min_distance_metres)
    error_checking.assert_is_geq(percentile_level, 0.)
    error_checking.assert_is_leq(percentile_level, 100.)

    min_distance_metres = rounder.round_to_nearest(
        min_distance_metres, DISTANCE_PRECISION_METRES)
    max_distance_metres = rounder.round_to_nearest(
        max_distance_metres, DISTANCE_PRECISION_METRES)
    percentile_level = rounder.round_to_nearest(
        percentile_level, PERCENTILE_LEVEL_PRECISION)

    return {
        MIN_LEAD_TIME_NAME: min_lead_time_sec,
        MAX_LEAD_TIME_NAME: max_lead_time_sec,
        MIN_DISTANCE_NAME: min_distance_metres,
        MAX_DISTANCE_NAME: max_distance_metres,
        PERCENTILE_LEVEL_NAME: percentile_level
    }


def _check_class_cutoffs(class_cutoffs_kt):
    """Error-checks (and if necessary, rounds) cutoffs for classification.

    C = number of classes
    c = C - 1 = number of class cutoffs

    :param class_cutoffs_kt: length-c numpy array of class cutoffs in knots
        (nautical miles per hour).
    :return: class_cutoffs_kt: Same as input, except with only unique rounded
        values.
    """

    class_cutoffs_kt = rounder.round_to_nearest(
        class_cutoffs_kt, CLASS_CUTOFF_PRECISION_KT)
    class_cutoffs_kt, _, _ = classifn_utils.classification_cutoffs_to_ranges(
        class_cutoffs_kt, non_negative_only=True)

    return class_cutoffs_kt


def column_name_to_label_params(column_name):
    """Determines parameters of label from column name.

    Label may be for either regression or classification.  If column name does
    not correspond to a label, this method will return None.

    :param column_name: Name of column.
    :return: parameter_dict: Dictionary with the following keys.
    parameter_dict['min_lead_time_sec']: See doc for _check_regression_params.
    parameter_dict['max_lead_time_sec']: See doc for _check_regression_params.
    parameter_dict['min_distance_metres']: See doc for _check_regression_params.
    parameter_dict['max_distance_metres']: See doc for _check_regression_params.
    parameter_dict['percentile_level']: See doc for _check_regression_params.
    parameter_dict['class_cutoffs_kt']: If learning goal is classification, this
        will be a numpy array (see documentation for _check_class_cutoffs).  If
        learning goal is regression, this will be None.
    """

    if column_name.startswith(PREFIX_FOR_REGRESSION_LABEL):
        is_goal_regression = True
        column_name = column_name[(len(PREFIX_FOR_REGRESSION_LABEL) + 1):]
    elif column_name.startswith(PREFIX_FOR_CLASSIFICATION_LABEL):
        is_goal_regression = False
        column_name = column_name[(len(PREFIX_FOR_CLASSIFICATION_LABEL) + 1):]
    else:
        return None

    column_name_parts = column_name.split('_')
    if is_goal_regression and len(column_name_parts) != 3:
        return None
    if not is_goal_regression and len(column_name_parts) != 4:
        return None

    percentile_part = column_name_parts[0]
    if not percentile_part.startswith('percentile='):
        return None

    percentile_part = percentile_part.replace('percentile=', '')
    try:
        percentile_level = float(percentile_part)
    except ValueError:
        return None

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
        min_distance_metres = float(int(distance_parts[0]))
        max_distance_metres = float(int(distance_parts[1]))
    except ValueError:
        return None

    if is_goal_regression:
        class_cutoffs_kt = None
    else:
        class_cutoff_part = column_name_parts[3]
        if not class_cutoff_part.startswith('cutoffs='):
            return None
        if not class_cutoff_part.endswith('kt'):
            return None

        class_cutoff_part = class_cutoff_part.replace('cutoffs=', '').replace(
            'kt', '')
        class_cutoff_parts = class_cutoff_part.split('-')

        try:
            class_cutoffs_kt = numpy.array(
                [int(c) for c in class_cutoff_parts]).astype(float)
        except ValueError:
            return None

    return {
        MIN_LEAD_TIME_NAME: min_lead_time_sec,
        MAX_LEAD_TIME_NAME: max_lead_time_sec,
        MIN_DISTANCE_NAME: min_distance_metres,
        MAX_DISTANCE_NAME: max_distance_metres,
        PERCENTILE_LEVEL_NAME: percentile_level,
        CLASS_CUTOFFS_NAME: class_cutoffs_kt
    }


def get_column_name_for_regression_label(
        min_lead_time_sec=None, max_lead_time_sec=None,
        min_distance_metres=None, max_distance_metres=None,
        percentile_level=None):
    """Generates column name for regression label.

    :param min_lead_time_sec: See documentation for label_wind_for_regression.
    :param max_lead_time_sec: See documentation for label_wind_for_regression.
    :param min_distance_metres: See documentation for label_wind_for_regression.
    :param max_distance_metres: See documentation for label_wind_for_regression.
    :param percentile_level: See documentation for label_wind_for_regression.
    :return: column_name: Column name for regression label.
    """

    parameter_dict = _check_regression_params(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_distance_metres=min_distance_metres,
        max_distance_metres=max_distance_metres,
        percentile_level=percentile_level)

    min_lead_time_sec = parameter_dict[MIN_LEAD_TIME_NAME]
    max_lead_time_sec = parameter_dict[MAX_LEAD_TIME_NAME]
    min_distance_metres = parameter_dict[MIN_DISTANCE_NAME]
    max_distance_metres = parameter_dict[MAX_DISTANCE_NAME]
    percentile_level = parameter_dict[PERCENTILE_LEVEL_NAME]

    return (
        '{0:s}_percentile={1:05.1f}_lead-time={2:04d}-{3:04d}sec_' +
        'distance={4:05d}-{5:05d}m').format(
            PREFIX_FOR_REGRESSION_LABEL, percentile_level, min_lead_time_sec,
            max_lead_time_sec, int(min_distance_metres),
            int(max_distance_metres))


def get_column_name_for_classification_label(
        min_lead_time_sec=None, max_lead_time_sec=None,
        min_distance_metres=None, max_distance_metres=None,
        percentile_level=None, class_cutoffs_kt=None):
    """Generates column name for regression label.

    :param min_lead_time_sec: See documentation for label_wind_for_regression.
    :param max_lead_time_sec: See documentation for label_wind_for_regression.
    :param min_distance_metres: See documentation for label_wind_for_regression.
    :param max_distance_metres: See documentation for label_wind_for_regression.
    :param percentile_level: See documentation for label_wind_for_regression.
    :param class_cutoffs_kt: See documentation for _check_class_cutoffs.
    :return: column_name: Column name for classification label.
    """

    column_name = get_column_name_for_regression_label(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_distance_metres=min_distance_metres,
        max_distance_metres=max_distance_metres,
        percentile_level=percentile_level)

    class_cutoffs_kt = _check_class_cutoffs(class_cutoffs_kt)
    num_cutoffs = len(class_cutoffs_kt)

    column_name = column_name.replace(
        PREFIX_FOR_REGRESSION_LABEL,
        PREFIX_FOR_CLASSIFICATION_LABEL) + '_cutoffs='

    for k in range(num_cutoffs):
        if k == 0:
            column_name += '{0:02d}'.format(int(class_cutoffs_kt[k]))
        else:
            column_name += '-{0:02d}'.format(int(class_cutoffs_kt[k]))

    return column_name + 'kt'


def get_regression_label_columns(storm_to_winds_table):
    """Returns names of columns with regression labels.

    :param storm_to_winds_table: pandas DataFrame.
    :return: regression_label_column_names: 1-D list containing names of columns
        with regression labels.  If there are no columns with regression labels,
        this is None.
    """

    column_names = list(storm_to_winds_table)
    regression_label_column_names = None

    for this_column_name in column_names:
        this_parameter_dict = column_name_to_label_params(this_column_name)
        if this_parameter_dict is None:
            continue
        if this_parameter_dict[CLASS_CUTOFFS_NAME] is not None:
            continue

        if regression_label_column_names is None:
            regression_label_column_names = [this_column_name]
        else:
            regression_label_column_names.append(this_column_name)

    return regression_label_column_names


def get_classification_label_columns(storm_to_winds_table):
    """Returns names of columns with classification labels.

    :param storm_to_winds_table: pandas DataFrame.
    :return: classification_label_column_names: 1-D list containing names of
        columns with classification labels.  If there are no columns with
        classification labels, this is None.
    """

    column_names = list(storm_to_winds_table)
    classification_label_column_names = None

    for this_column_name in column_names:
        this_parameter_dict = column_name_to_label_params(this_column_name)
        if this_parameter_dict is None:
            continue
        if this_parameter_dict[CLASS_CUTOFFS_NAME] is None:
            continue

        if classification_label_column_names is None:
            classification_label_column_names = [this_column_name]
        else:
            classification_label_column_names.append(this_column_name)

    return classification_label_column_names


def get_label_columns(storm_to_winds_table):
    """Returns names of columns with regression or classification labels.

    :param storm_to_winds_table: pandas DataFrame.
    :return: label_column_names: 1-D list containing names of columns with
        regression or classification labels.  If there are no columns with
        labels, this is None.
    """

    label_column_names = []

    regression_label_column_names = get_regression_label_columns(
        storm_to_winds_table)
    if regression_label_column_names is not None:
        label_column_names += regression_label_column_names

    classification_label_column_names = get_classification_label_columns(
        storm_to_winds_table)
    if classification_label_column_names is not None:
        label_column_names += classification_label_column_names

    if not label_column_names:
        return None
    return label_column_names


def check_label_table(label_table, require_storm_objects=True):
    """Ensures that pandas DataFrame contains labels.

    :param label_table: pandas DataFrame.
    :param require_storm_objects: Boolean flag.  If True, label_table must
        contain columns "storm_id" and "unix_time_sec".  If False, label_table
        does not need these columns.
    :return: label_column_names: 1-D list containing names of columns with
        regression or classification labels.
    :raises: ValueError: if label_table does not contain any columns with
        regression or classification labels.
    """

    label_column_names = get_label_columns(label_table)
    if label_column_names is None:
        raise ValueError(
            'label_table does not contain any column with regression or '
            'classification labels.')

    if require_storm_objects:
        error_checking.assert_columns_in_dataframe(
            label_table, MANDATORY_COLUMNS)

    return label_column_names


def label_wind_for_regression(
        storm_to_winds_table, min_lead_time_sec=DEFAULT_MIN_LEAD_TIME_SEC,
        max_lead_time_sec=DEFAULT_MAX_LEAD_TIME_SEC,
        min_distance_metres=DEFAULT_MIN_DISTANCE_METRES,
        max_distance_metres=DEFAULT_MAX_DISTANCE_METRES,
        percentile_level=DEFAULT_PERCENTILE_LEVEL):
    """Labels each storm object for regression.

    t = valid time for a given storm object

    Each label is based on wind observations linked to the storm object,
    occurring at times (t + min_lead_time_sec)...(t + max_lead_time_sec) and
    distances of min_distance_metres...max_distance_metres from the boundary of
    the storm object.  Linkages (association of storm objects with wind
    observations) are created by storm_to_winds.py.

    :param storm_to_winds_table: pandas DataFrame with columns documented in
        `storm_to_winds.write_storm_to_winds_table`.
    :param min_lead_time_sec: Minimum lead time (wind time minus storm-object
        time).  Wind observations occurring before t + min_lead_time_sec are
        ignored.
    :param max_lead_time_sec: Maximum lead time (wind time minus storm-object
        time).  Wind observations occurring after t + max_lead_time_sec are
        ignored.
    :param min_distance_metres: Minimum distance between storm boundary and wind
        observations.  Wind observations nearer to the storm are ignored.
    :param max_distance_metres: Maximum distance between storm boundary and wind
        observations.  Wind observations farther from the storm are ignored.
    :param percentile_level: The label for each storm object will be the [q]th-
        percentile speed of wind observations in the given time and distance
        ranges, where q = percentile_level.
    :return: storm_to_winds_table: Same as input, except for the following.
        [1] May have fewer rows (storm objects occurring too close to end of
            tracking period are removed).
        [2] Contains additional column with regression labels.  The name of this
            column is determined by get_column_name_for_regression_label.
        [3] Contains additional column "num_observations_for_label", with number
            of observations used to create each regression label.
    """

    parameter_dict = _check_regression_params(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_distance_metres=min_distance_metres,
        max_distance_metres=max_distance_metres,
        percentile_level=percentile_level)

    min_lead_time_sec = parameter_dict[MIN_LEAD_TIME_NAME]
    max_lead_time_sec = parameter_dict[MAX_LEAD_TIME_NAME]
    min_distance_metres = parameter_dict[MIN_DISTANCE_NAME]
    max_distance_metres = parameter_dict[MAX_DISTANCE_NAME]
    percentile_level = parameter_dict[PERCENTILE_LEVEL_NAME]

    times_before_end_of_tracking_sec = (
        storm_to_winds_table[tracking_io.TRACKING_END_TIME_COLUMN] -
        storm_to_winds_table[tracking_io.TIME_COLUMN])
    bad_storm_object_rows = numpy.where(
        times_before_end_of_tracking_sec < max_lead_time_sec)[0]
    storm_to_winds_table.drop(
        storm_to_winds_table.index[bad_storm_object_rows], axis=0, inplace=True)

    num_storm_objects = len(storm_to_winds_table.index)
    labels_m_s01 = numpy.full(num_storm_objects, numpy.nan)

    for i in range(num_storm_objects):
        these_relative_wind_times_sec = storm_to_winds_table[
            storm_to_winds.RELATIVE_TIMES_COLUMN].values[i]
        these_distances_metres = storm_to_winds_table[
            storm_to_winds.LINKAGE_DISTANCES_COLUMN].values[i]

        these_valid_time_flags = numpy.logical_and(
            these_relative_wind_times_sec >= min_lead_time_sec,
            these_relative_wind_times_sec <= max_lead_time_sec)
        these_valid_distance_flags = numpy.logical_and(
            these_distances_metres >= min_distance_metres,
            these_distances_metres <= max_distance_metres)
        these_valid_wind_flags = numpy.logical_and(
            these_valid_time_flags, these_valid_distance_flags)
        if not numpy.any(these_valid_wind_flags):
            labels_m_s01[i] = 0.
            continue

        these_valid_wind_indices = numpy.where(these_valid_wind_flags)[0]
        these_u_winds_m_s01 = storm_to_winds_table[
            storm_to_winds.U_WINDS_COLUMN].values[i][these_valid_wind_indices]
        these_v_winds_m_s01 = storm_to_winds_table[
            storm_to_winds.V_WINDS_COLUMN].values[i][these_valid_wind_indices]

        these_wind_speeds_m_s01 = numpy.sqrt(
            these_u_winds_m_s01 ** 2 + these_v_winds_m_s01 ** 2)
        labels_m_s01[i] = numpy.percentile(
            these_wind_speeds_m_s01, percentile_level)

    label_column_name = get_column_name_for_regression_label(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_distance_metres=min_distance_metres,
        max_distance_metres=max_distance_metres,
        percentile_level=percentile_level)

    return storm_to_winds_table.assign(**{label_column_name: labels_m_s01})


def label_wind_for_classification(
        storm_to_winds_table, min_lead_time_sec=DEFAULT_MIN_LEAD_TIME_SEC,
        max_lead_time_sec=DEFAULT_MAX_LEAD_TIME_SEC,
        min_distance_metres=DEFAULT_MIN_DISTANCE_METRES,
        max_distance_metres=DEFAULT_MAX_DISTANCE_METRES,
        percentile_level=DEFAULT_PERCENTILE_LEVEL,
        class_cutoffs_kt=DEFAULT_CLASS_CUTOFFS_KT):
    """Labels each storm object for classification.

    :param storm_to_winds_table: See documentation for
        label_wind_for_regression.
    :param min_lead_time_sec: See documentation for label_wind_for_regression.
    :param max_lead_time_sec: See documentation for label_wind_for_regression.
    :param min_distance_metres: See documentation for label_wind_for_regression.
    :param max_distance_metres: See documentation for label_wind_for_regression.
    :param percentile_level: See documentation for label_wind_for_regression.
    :param class_cutoffs_kt: See documentation for _check_class_cutoffs.
    :return: storm_to_winds_table: Same as input, but with 3 additional columns
        (one containing classification labels, one containing corresponding
        regression labels).  The names of these columns are given by
        get_column_name_for_classification_label.  The other is listed below.
    storm_to_winds_table.num_observations_for_label: Number of wind observations
        used to create label.
    """

    parameter_dict = _check_regression_params(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_distance_metres=min_distance_metres,
        max_distance_metres=max_distance_metres,
        percentile_level=percentile_level)

    min_lead_time_sec = parameter_dict[MIN_LEAD_TIME_NAME]
    max_lead_time_sec = parameter_dict[MAX_LEAD_TIME_NAME]
    min_distance_metres = parameter_dict[MIN_DISTANCE_NAME]
    max_distance_metres = parameter_dict[MAX_DISTANCE_NAME]
    percentile_level = parameter_dict[PERCENTILE_LEVEL_NAME]

    class_cutoffs_m_s01 = KT_TO_METRES_PER_SECOND * _check_class_cutoffs(
        class_cutoffs_kt)

    storm_to_winds_table = label_wind_for_regression(
        storm_to_winds_table, min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_distance_metres=min_distance_metres,
        max_distance_metres=max_distance_metres,
        percentile_level=percentile_level)

    regression_label_column_name = get_column_name_for_classification_label(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_distance_metres=min_distance_metres,
        max_distance_metres=max_distance_metres,
        percentile_level=percentile_level)
    regression_labels_m_s01 = storm_to_winds_table[
        regression_label_column_name].values

    storm_classes = classifn_utils.classify_values(
        regression_labels_m_s01, class_cutoffs=class_cutoffs_m_s01,
        non_negative_only=True)

    classification_label_column_name = get_column_name_for_classification_label(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_distance_metres=min_distance_metres,
        max_distance_metres=max_distance_metres,
        percentile_level=percentile_level, class_cutoffs_kt=class_cutoffs_kt)

    return storm_to_winds_table.assign(
        **{classification_label_column_name: storm_classes})


def write_labels(storm_to_winds_table, pickle_file_name):
    """Writes labels for storm objects to a Pickle file.

    :param storm_to_winds_table: pandas DataFrame created by
        label_wind_for_regression or label_wind_for_classification.
    :param pickle_file_name: Path to output file.
    :raises: ValueError: if storm_to_winds_table does not contain any columns
        with a regression or classification label.
    """

    label_column_names = check_label_table(
        storm_to_winds_table, require_storm_objects=True)
    columns_to_write = storm_to_winds.COLUMNS_TO_WRITE + label_column_names

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(storm_to_winds_table[columns_to_write], pickle_file_handle)
    pickle_file_handle.close()


def read_labels(pickle_file_name):
    """Reads labels for storm objects from a Pickle file.

    :param pickle_file_name: Path to input file.
    :return: storm_to_winds_table: pandas DataFrame with columns documented in
        label_wind_for_regression or label_wind_for_classification.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    storm_to_winds_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    check_label_table(storm_to_winds_table, require_storm_objects=True)
    return storm_to_winds_table
