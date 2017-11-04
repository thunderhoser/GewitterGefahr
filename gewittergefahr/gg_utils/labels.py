"""Labels each storm object for machine learning.

--- DEFINITIONS ---

Label = "dependent variable" = "target variable" = "outcome" = "predictand".
This is the variable that machine learning is trying to predict.  An example is
the max wind speed associated with the storm object.
"""

import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.linkage import storm_to_winds
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import error_checking

KT_TO_METRES_PER_SECOND = 1.852 / 3.6

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
CLASS_MINIMA_NAME = 'class_minima_kt'
CLASS_MAXIMA_NAME = 'class_maxima_kt'

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
    q = percentile level
    U_q = [q]th-percentile wind speed linked to storm object

    :param class_cutoffs_kt: length-c numpy array of class cutoffs in knots
        (nautical miles per hour).  Storm objects with
        U_q in [0, class_cutoffs_kt[0]) will be in class 0; storm objects with
        U_q in [class_cutoffs_kt[k - 1], class_cutoffs_kt[k]) will be in class
        k; and storm objects with U_q >= class_cutoffs_kt[-1] will be in the
        highest class, C - 1.
    :return: parameter_dict: Dictionary with the following keys.
    parameter_dict['class_cutoffs_kt']: Same as input, but maybe rounded.
    parameter_dict['class_minima_kt']: length-C numpy array of class minima.
    parameter_dict['class_maxima_kt']: length-C numpy array of class maxima.
    """

    error_checking.assert_is_greater_numpy_array(class_cutoffs_kt, 0.)
    error_checking.assert_is_numpy_array(class_cutoffs_kt, num_dimensions=1)

    class_cutoffs_kt = numpy.sort(numpy.unique(rounder.round_to_nearest(
        class_cutoffs_kt, CLASS_CUTOFF_PRECISION_KT)))

    num_classes = len(class_cutoffs_kt) + 1
    class_minima_kt = numpy.full(num_classes, numpy.nan)
    class_maxima_kt = numpy.full(num_classes, numpy.nan)

    for k in range(num_classes):
        if k == 0:
            class_minima_kt[k] = 0.
            class_maxima_kt[k] = class_cutoffs_kt[k]
        elif k == num_classes - 1:
            class_minima_kt[k] = class_cutoffs_kt[k - 1]
            class_maxima_kt[k] = numpy.inf
        else:
            class_minima_kt[k] = class_cutoffs_kt[k - 1]
            class_maxima_kt[k] = class_cutoffs_kt[k]

    return {
        CLASS_CUTOFFS_NAME: class_cutoffs_kt,
        CLASS_MINIMA_NAME: class_minima_kt, CLASS_MAXIMA_NAME: class_maxima_kt
    }


def _classify_wind_speeds(wind_speeds_m_s01, class_minima_m_s01=None,
                          class_maxima_m_s01=None):
    """Determines class for each wind speed.

    N = number of wind speeds
    C = number of classes

    :param wind_speeds_m_s01: length-N numpy array of wind speeds (metres per
        second).
    :param class_minima_m_s01: length-C numpy array of class minima (metres per
        second).
    :param class_maxima_m_s01: length-C numpy array of class maxima (metres per
        second).
    :return: wind_classes: length-N numpy array of ordinal class numbers.  For
        example, if wind_classes[i] = k, this means the [i]th wind speed belongs
        to the [k]th class.  Ordinal numbers are zero-based, so all values in
        wind_classes are from 0...(C - 1).
    """

    num_observations = len(wind_speeds_m_s01)
    wind_classes = numpy.full(num_observations, -1, dtype=int)
    num_classes = len(class_minima_m_s01)

    for k in range(num_classes):
        these_flags = numpy.logical_and(
            wind_speeds_m_s01 >= class_minima_m_s01[k],
            wind_speeds_m_s01 < class_maxima_m_s01[k])
        these_indices = numpy.where(these_flags)[0]
        wind_classes[these_indices] = k

    return wind_classes


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

    parameter_dict = _check_class_cutoffs(class_cutoffs_kt)
    class_cutoffs_kt = parameter_dict[CLASS_CUTOFFS_NAME]
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
    :return: storm_to_winds_table: Same as input, with two exceptions.
        [1] May have fewer rows (storm objects occurring too close to end of
            tracking period are removed).
        [2] Contains additional column with regression labels.  The name of this
            column is determined by get_column_name_for_regression_label.
    :return: labels_m_s01: 1-D numpy array of regression labels (metres per
        second).  This is the same as the additional column in
        storm_to_winds_table.
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

    return (storm_to_winds_table.assign(**{label_column_name: labels_m_s01}),
            labels_m_s01)


def label_wind_for_classification(
        storm_to_winds_table, min_lead_time_sec=DEFAULT_MIN_LEAD_TIME_SEC,
        max_lead_time_sec=DEFAULT_MAX_LEAD_TIME_SEC,
        min_distance_metres=DEFAULT_MIN_DISTANCE_METRES,
        max_distance_metres=DEFAULT_MAX_DISTANCE_METRES,
        percentile_level=DEFAULT_PERCENTILE_LEVEL,
        class_cutoffs_kt=DEFAULT_CLASS_CUTOFFS_KT):
    """Labels each storm object for classification.

    :param storm_to_winds_table: See documentation for label_wind_for_regression.
    :param min_lead_time_sec: See documentation for label_wind_for_regression.
    :param max_lead_time_sec: See documentation for label_wind_for_regression.
    :param min_distance_metres: See documentation for label_wind_for_regression.
    :param max_distance_metres: See documentation for label_wind_for_regression.
    :param percentile_level: See documentation for label_wind_for_regression.
    :param class_cutoffs_kt: See documentation for _check_class_cutoffs.
    :return: storm_to_winds_table: Same as input, but with one additional column
        containing classification labels.  The name of this column is given by
        get_column_name_for_classification_label.
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

    parameter_dict = _check_class_cutoffs(class_cutoffs_kt)
    class_cutoffs_kt = parameter_dict[CLASS_CUTOFFS_NAME]
    class_minima_m_s01 = KT_TO_METRES_PER_SECOND * parameter_dict[
        CLASS_MINIMA_NAME]
    class_maxima_m_s01 = KT_TO_METRES_PER_SECOND * parameter_dict[
        CLASS_MAXIMA_NAME]

    storm_to_winds_table, regression_labels_m_s01 = label_wind_for_regression(
        storm_to_winds_table, min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_distance_metres=min_distance_metres,
        max_distance_metres=max_distance_metres,
        percentile_level=percentile_level)

    storm_classes = _classify_wind_speeds(
        regression_labels_m_s01, class_minima_m_s01=class_minima_m_s01,
        class_maxima_m_s01=class_maxima_m_s01)

    label_column_name = get_column_name_for_classification_label(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_distance_metres=min_distance_metres,
        max_distance_metres=max_distance_metres,
        percentile_level=percentile_level, class_cutoffs_kt=class_cutoffs_kt)

    return storm_to_winds_table.assign(**{label_column_name: storm_classes})
