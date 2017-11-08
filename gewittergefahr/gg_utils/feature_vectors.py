"""Methods for handling feature vectors.

--- GENERAL DEFINITIONS ---

Feature (or "attribute" or "predictor variable" or "independent variable") =
variable used to predict the label

Label (or "outcome" or "dependent variable" or "target variable" or
"predictand") = variable to be predicted.

Feature vector = list of all features for one example (storm object).

--- DEFINITIONS FOR SAMPLING METHODS ---

q = percentile level used to create both regression and classification labels.
    For more on the role of q, see `labels.label_wind_for_regression` and
    `labels.label_wind_for_classification`.

U_q = [q]th percentile of all wind speeds linked to a given storm object.

N_obs = number of wind observations linked to a given storm object.

T = range of lead times used to create label.

Live storm object = storm object for which the corresponding cell exists at some
                    point during T.

Dead storm object = storm object for which the corresponding cell no longer
                    exists during T.

--- SAMPLING METHODS ---

[1] "uniform_wind_speed": Storm objects are sampled uniformly with respect to
    U_q.
[2] "min_observations": All storm objects with N_obs >= threshold are used.
    Also, dead storm objects are sampled randomly to preserve the ratio of live
    to dead storm objects.  This helps to eliminate survivor bias, where only
    storms with a minimum remaining lifetime (which are more likely to produce
    severe wind) are used.
[3] "min_observations_plus": Same as "min_observations", except that all storm
    objects in the highest class (i.e., with the highest possible classification
    label) are used.
[4] "min_obs_density": All storm objects with observation density >= threshold
    are used.  Also, dead storm objects are sampled randomly to preserve the
    ratio of live to dead storm objects.  Observation density is a spatial
    density (number per m^2), where the denominator is the area of the storm-
    centered buffer used to create the label.  For example, if the label is
    based on wind observations from 5-10 km outside the storm, the denominator
    is the area of the 5--10-km buffer around the storm.
[5] "min_obs_density_plus": Same as "min_obs_density", except that all storm
    objects in the highest class (i.e., with the highest possible classification
    label) are used.
"""

import os.path
import pickle
import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import radar_statistics as radar_stats
from gewittergefahr.gg_utils import shape_statistics as shape_stats
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import classification_utils as classifn_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.linkage import storm_to_winds

TOLERANCE = 1e-6
KT_TO_METRES_PER_SECOND = 1.852 / 3.6

FEATURE_FILE_PREFIX = 'features'
FEATURE_FILE_EXTENSION = '.p'
TIME_FORMAT_IN_FILE_NAMES = '%Y-%m-%d-%H%M%S'

STORM_TO_WIND_COLUMNS_TO_KEEP = [
    tracking_io.STORM_ID_COLUMN, tracking_io.TIME_COLUMN,
    tracking_io.POLYGON_OBJECT_LATLNG_COLUMN, storm_to_winds.END_TIME_COLUMN,
    labels.NUM_OBSERVATIONS_FOR_LABEL_COLUMN]
COLUMNS_TO_MERGE_ON = [tracking_io.STORM_ID_COLUMN, tracking_io.TIME_COLUMN]
INPUT_COLUMNS_TO_MAKE_BUFFERS = (
    COLUMNS_TO_MERGE_ON + [tracking_io.POLYGON_OBJECT_LATLNG_COLUMN])

LIVE_SELECTED_INDICES_KEY = 'live_selected_indices'
LIVE_INDICES_KEY = 'live_indices'
DEAD_INDICES_KEY = 'dead_indices'
SPEED_CATEGORY_KEY_FOR_UNIFORM_SAMPLING = 'speed_category_by_storm_object'
NUM_OBSERVATIONS_KEY = 'num_observations_by_storm_object'

UNIFORM_SAMPLING_METHOD = 'uniform_wind_speed'
MIN_OBS_SAMPLING_METHOD = 'min_observations'
MIN_OBS_PLUS_SAMPLING_METHOD = 'min_observations_plus'
MIN_DENSITY_SAMPLING_METHOD = 'min_obs_density'
MIN_DENSITY_PLUS_SAMPLING_METHOD = 'min_obs_density_plus'

SAMPLING_METHODS = [
    UNIFORM_SAMPLING_METHOD, MIN_OBS_SAMPLING_METHOD,
    MIN_OBS_PLUS_SAMPLING_METHOD, MIN_DENSITY_SAMPLING_METHOD,
    MIN_DENSITY_PLUS_SAMPLING_METHOD]
NON_UNIFORM_SAMPLING_METHODS = [
    MIN_OBS_SAMPLING_METHOD, MIN_OBS_PLUS_SAMPLING_METHOD,
    MIN_DENSITY_SAMPLING_METHOD, MIN_DENSITY_PLUS_SAMPLING_METHOD]

DEFAULT_MIN_OBSERVATIONS_FOR_SAMPLING = 25
DEFAULT_MIN_OBS_DENSITY_FOR_SAMPLING_M02 = 1e-7
DEFAULT_CUTOFFS_FOR_UNIFORM_SAMPLING_M_S01 = (
    KT_TO_METRES_PER_SECOND * numpy.array([10., 20., 30., 40., 50.]))


def _find_live_and_dead_storms(feature_table):
    """Finds live and dead storm objects.

    :param feature_table: pandas DataFrame created by
        join_features_and_label_for_storm_objects.  Each row is one storm
        object.
    :return: live_indices: 1-D numpy array with indices (rows) of live storm
        objects.
    :return: dead_indices: 1-D numpy array with indices (rows) of dead storm
        objects.
    """

    _, regression_label_column_name, _ = check_feature_table(
        feature_table, require_storm_objects=True)
    label_parameter_dict = labels.column_name_to_label_params(
        regression_label_column_name)
    min_lead_time_sec = label_parameter_dict[labels.MIN_LEAD_TIME_NAME]

    remaining_storm_lifetimes_sec = (
        feature_table[storm_to_winds.END_TIME_COLUMN].values -
        feature_table[tracking_io.TIME_COLUMN].values)
    live_flags = remaining_storm_lifetimes_sec >= min_lead_time_sec

    return (numpy.array(numpy.where(live_flags)[0]),
            numpy.array(numpy.where(numpy.invert(live_flags))[0]))


def _select_dead_storms(live_indices=None, dead_indices=None,
                        live_selected_indices=None):
    """Selects dead storm objects.

    f_dead = fraction of dead storm objects

    This method randomly selects dead storm objects, so that the f_dead in the
    selected set = f_dead in the full set.

    :param live_indices: 1-D numpy array with indices of live storm objects.
    :param dead_indices: 1-D numpy array with indices of dead storm objects.
    :param live_selected_indices: 1-D numpy array with indices of live storm
        objects that have already been selected.
    :return: selected_indices: 1-D numpy array with indices of all storm objects
        (alive and dead) selected.
    """

    dead_storm_fraction = float(len(dead_indices)) / (
        len(dead_indices) + len(live_indices))

    num_live_storms_selected = len(live_selected_indices)
    num_dead_storms_to_select = int(numpy.round(
        num_live_storms_selected * dead_storm_fraction / (
            1 - dead_storm_fraction)))

    dead_selected_indices = numpy.random.choice(
        dead_indices, size=num_dead_storms_to_select, replace=False)
    return numpy.sort(numpy.concatenate((
        live_selected_indices, dead_selected_indices)))


def _select_storms_uniformly_by_category(category_by_storm_object,
                                         num_observations_by_storm_object):
    """Selects an equal number of storm objects from each category.

    Within each category, this method selects the storm objects with the
    greatest numbers of attached observations.  In other words, the best-
    observed storm objects are selected in each category.

    N = number of storm objects

    :param category_by_storm_object: length-N numpy array with integer category
        for each storm object.
    :param num_observations_by_storm_object: length-N numpy array with number of
        observations attached to each storm object.
    :return: selected_indices: 1-D numpy array with indices of all selected
        storm objects.
    """

    unique_categories, orig_to_unique_category_indices = numpy.unique(
        category_by_storm_object, return_inverse=True)

    num_unique_categories = len(unique_categories)
    num_storm_objects_by_category = numpy.full(
        num_unique_categories, 0, dtype=int)

    for k in range(num_unique_categories):
        these_storm_indices = numpy.where(
            orig_to_unique_category_indices == k)[0]
        num_storm_objects_by_category[k] = len(these_storm_indices)

    min_storm_objects_in_category = numpy.min(num_storm_objects_by_category)
    selected_indices = numpy.array([])

    for k in range(num_unique_categories):
        these_storm_indices = numpy.where(
            orig_to_unique_category_indices == k)[0]
        these_sort_indices = numpy.argsort(
            -num_observations_by_storm_object[these_storm_indices])

        these_selected_indices = (
            these_sort_indices[:min_storm_objects_in_category])
        these_selected_indices = these_storm_indices[these_selected_indices]
        selected_indices = numpy.concatenate((
            selected_indices, these_selected_indices))

    return numpy.sort(selected_indices)


def _get_observation_densities(feature_table):
    """Finds wind-observation density in buffer around each storm object.

    Specifically, for each storm object, finds observation density in buffer
    used to create regression or classification labels.

    N = number of storm objects

    :param feature_table: N-row pandas DataFrame created by
        join_features_and_label_for_storm_objects.
    :return: feature_table: Same as input, except that columns may have been
        added by `storm_tracking_io.make_buffers_around_polygons`.
    :return: observation_densities_m02: length-N numpy array of observation
        densities (number per m^2).
    """

    _, regression_label_column_name, _, = check_feature_table(
        feature_table, require_storm_objects=True)
    label_parameter_dict = labels.column_name_to_label_params(
        regression_label_column_name)

    min_buffer_distance_metres = label_parameter_dict[labels.MIN_DISTANCE_NAME]
    max_buffer_distance_metres = label_parameter_dict[labels.MAX_DISTANCE_NAME]
    if min_buffer_distance_metres < TOLERANCE:
        min_buffer_distance_metres = numpy.nan

    buffer_column_name = tracking_io.distance_buffer_to_column_name(
        min_buffer_distance_metres, max_buffer_distance_metres)
    if buffer_column_name not in feature_table:
        buffer_table = feature_table[INPUT_COLUMNS_TO_MAKE_BUFFERS]
        buffer_table = tracking_io.make_buffers_around_polygons(
            buffer_table,
            min_buffer_dists_metres=numpy.array([min_buffer_distance_metres]),
            max_buffer_dists_metres=numpy.array([max_buffer_distance_metres]))

        feature_table = feature_table.merge(
            buffer_table, on=COLUMNS_TO_MERGE_ON, how='inner')

    num_storm_objects = len(feature_table.index)
    buffer_areas_m2 = numpy.full(num_storm_objects, numpy.nan)
    for i in range(num_storm_objects):
        this_polygon_object_xy, _ = polygons.project_latlng_to_xy(
            feature_table[buffer_column_name].values[i])
        buffer_areas_m2[i] = this_polygon_object_xy.area

    observation_densities_m02 = (
        feature_table[labels.NUM_OBSERVATIONS_FOR_LABEL_COLUMN].values /
        buffer_areas_m2)
    return feature_table, observation_densities_m02


def _indices_from_file_specific_to_overall(indices_by_file,
                                           num_objects_by_file):
    """Converts indices from file-specific to overall.

    For example, in a one-based indexing system: if file 1 contains 50 objects
    and file 2 contains 100 objects, object 3 in file 3 will be object 153
    overall (50 + 100 + 3).

    In a zero-based indexing system: if file 1 contains 50 objects
    and file 2 contains 100 objects, object 3 in file 3 will also be object 153
    overall (50 + 100 + [3 + 1] - 1).

    N = number of files

    :param indices_by_file: length-N list, where each element is a 1-D numpy
        array of object indices.
    :param num_objects_by_file: 1-D numpy array with number of objects in each
        file.
    :return: overall_indices: 1-D numpy array of overall object indices.
    """

    # TODO(thunderhoser): This method is more general than selecting feature
    # vectors.  Should go in a different file.

    overall_indices = numpy.array([])
    num_files = len(indices_by_file)

    for i in range(num_files):
        if i == 0:
            these_overall_indices = indices_by_file[i]
        else:
            these_overall_indices = (
                indices_by_file[i] + numpy.sum(num_objects_by_file[:i]))

        overall_indices = numpy.concatenate((
            overall_indices, these_overall_indices))

    return overall_indices


def _indices_from_overall_to_file_specific(overall_indices,
                                           num_objects_by_file):
    """Converts indices from overall to file-specific.

    This method is the inverse of _indices_from_file_specific_to_overall.

    N = number of files

    :param overall_indices: 1-D numpy array of overall indices.
    :param num_objects_by_file: 1-D numpy array with number of objects in each
        file.
    :return: indices_by_file: length-N list, where each element is a 1-D numpy
        array of object indices.
    """

    # TODO(thunderhoser): This method is more general than selecting feature
    # vectors.  Should go in a different file.

    num_files = len(num_objects_by_file)
    indices_by_file = [None] * num_files

    for i in range(num_files):
        if num_objects_by_file[i] == 0:
            indices_by_file[i] = numpy.array([], dtype=int)
            continue

        if i == 0:
            this_min_overall_index = 0
        else:
            this_min_overall_index = numpy.sum(num_objects_by_file[:i])

        this_max_overall_index = (
            this_min_overall_index + num_objects_by_file[i] - 1)
        this_file_flags = numpy.logical_and(
            overall_indices >= this_min_overall_index,
            overall_indices <= this_max_overall_index)
        this_file_indices = numpy.where(this_file_flags)[0]
        indices_by_file[i] = overall_indices[this_file_indices] - numpy.sum(
            num_objects_by_file[:i])

    return indices_by_file


def check_feature_table(feature_table, require_storm_objects=True):
    """Ensures that pandas DataFrame contains features and labels.

    feature_table must contain one or more feature columns.
    feature_table must contain either 1 or 2 label columns.  If 2 columns, there
    must be one regression label L_r and one classification label L_c, where L_r
    is the regression version of L_c.

    :param feature_table: pandas DataFrame.
    :param require_storm_objects: Boolean flag.  If True, feature_table must
        contain columns "storm_id" and "unix_time_sec".  If False, feature_table
        does not need these columns.
    :return: feature_column_names: 1-D list containing names of columns with
        features.
    :return: regression_label_column_name: Name of column with regression label.
        If there is no regression label, this will be None.
    :return: classification_label_column_name: Name of column with
        classification label.  If there is no regression label, this will be
        None.
    :raises: ValueError: if feature_table does not contain any feature columns.
    :raises: ValueError: if feature_table does not contain exactly one label
        column.
    """

    feature_column_names = radar_stats.get_statistic_columns(feature_table)

    shape_stat_column_names = shape_stats.get_statistic_columns(feature_table)
    if shape_stat_column_names:
        if feature_column_names:
            feature_column_names += shape_stat_column_names
        else:
            feature_column_names = shape_stat_column_names

    sounding_index_column_names = soundings.get_sounding_index_columns(
        feature_table)
    if sounding_index_column_names:
        if feature_column_names:
            feature_column_names += sounding_index_column_names
        else:
            feature_column_names = sounding_index_column_names

    if feature_column_names is None:
        raise ValueError(
            'feature_table does not contain any columns with features '
            '(predictor variables).')

    regression_label_column_names = labels.get_regression_label_columns(
        feature_table)
    if regression_label_column_names and len(
            regression_label_column_names) == 1:
        regression_label_column_name = regression_label_column_names[0]
    else:
        regression_label_column_name = None

    classification_label_column_names = labels.get_classification_label_columns(
        feature_table)
    if classification_label_column_names and len(
            classification_label_column_names) == 1:
        classification_label_column_name = classification_label_column_names[0]
    else:
        classification_label_column_name = None

    if regression_label_column_name and classification_label_column_name:
        classification_param_dict = labels.column_name_to_label_params(
            classification_label_column_name)

        this_regression_label_column_name = (
            labels.get_column_name_for_regression_label(
                min_lead_time_sec=classification_param_dict[
                    labels.MIN_LEAD_TIME_NAME],
                max_lead_time_sec=classification_param_dict[
                    labels.MAX_LEAD_TIME_NAME],
                min_distance_metres=classification_param_dict[
                    labels.MIN_DISTANCE_NAME],
                max_distance_metres=classification_param_dict[
                    labels.MAX_DISTANCE_NAME],
                percentile_level=classification_param_dict[
                    labels.PERCENTILE_LEVEL_NAME]))

        if this_regression_label_column_name != regression_label_column_name:
            regression_label_column_name = None
            classification_label_column_name = None

    if not (regression_label_column_name or classification_label_column_name):
        error_string = (
            '\n\n' + str(regression_label_column_names) +
            str(classification_label_column_names) + '\n\nfeature_table ' +
            'should contain one regression-label column, one classification-'
            'label column, or a classification-label column with the '
            'corresponding regression-label column.  Instead, contains label '
            'columns listed above.')
        raise ValueError(error_string)

    if require_storm_objects:
        error_checking.assert_columns_in_dataframe(
            feature_table, STORM_TO_WIND_COLUMNS_TO_KEEP)

    return (feature_column_names, regression_label_column_name,
            classification_label_column_name)


def join_features_and_label_for_storm_objects(
        radar_statistic_table=None, shape_statistic_table=None,
        sounding_index_table=None, storm_to_winds_table=None,
        label_column_name=None):
    """Joins tables with features and label.

    storm_to_winds_table may contain several columns with labels (for both
    classification and regression).  However, only one label will be joined with
    the other tables.

    :param radar_statistic_table: pandas DataFrame with radar statistics.
        Columns are documented in
        `radar_statistics.write_stats_for_storm_objects`.
    :param shape_statistic_table: pandas DataFrame with shape statistics.
        Columns are documented in
        `shape_statistics.write_stats_for_storm_objects`.
    :param sounding_index_table: pandas DataFrame with sounding indices.
        Columns are documented in
        `soundings.write_sounding_indices_for_storm_objects`.
    :param storm_to_winds_table: pandas DataFrame with labels.  Columns are
        documented in `storm_to_winds.write_storm_to_winds_table`.
    :param label_column_name: Name of label (in storm_to_winds_table) to be
        joined with the other tables.
    :return: feature_table: pandas DataFrame containing all columns with radar
        statistics, shape statistics, sounding indices, and the desired label
        (see input argument `label_column_name`).  If `label_column_name` is a
        classification label, feature_table will also contain the corresponding
        regression label.  If storm_to_winds_table contains distance buffers
        (with column names given by
        `storm_tracking_io.distance_buffer_to_column_name`), these will also be
        in feature_table.  Each row is one storm object.  Mandatory columns are
        listed below.
    feature_table.storm_id: String ID for storm cell.
    feature_table.unix_time_sec: Valid time of storm object.
    feature_table.end_time_unix_sec: End time of corresponding storm cell.
    feature_table.num_observations_for_label: Number of observations used to
        create regression and/or classification label for this storm object.
    feature_table.polygon_object_latlng: Instance of `shapely.geometry.Polygon`,
        defining the outline of the storm object.
    """

    feature_table = None

    if radar_statistic_table is not None:
        radar_stat_column_names = radar_stats.check_statistic_table(
            radar_statistic_table, require_storm_objects=True)
        feature_table = radar_statistic_table[
            COLUMNS_TO_MERGE_ON + radar_stat_column_names]

    if shape_statistic_table is not None:
        shape_stat_column_names = shape_stats.check_statistic_table(
            shape_statistic_table, require_storm_objects=True)
        shape_statistic_table = shape_statistic_table[
            COLUMNS_TO_MERGE_ON + shape_stat_column_names]

        if feature_table is None:
            feature_table = shape_statistic_table
        else:
            feature_table = feature_table.merge(
                shape_statistic_table, on=COLUMNS_TO_MERGE_ON, how='inner')

    if sounding_index_table is not None:
        sounding_index_column_names = soundings.check_sounding_index_table(
            sounding_index_table, require_storm_objects=True)
        sounding_index_table = sounding_index_table[
            COLUMNS_TO_MERGE_ON + sounding_index_column_names]

        if feature_table is None:
            feature_table = sounding_index_table
        else:
            feature_table = feature_table.merge(
                sounding_index_table, on=COLUMNS_TO_MERGE_ON, how='inner')

    label_parameter_dict = labels.column_name_to_label_params(label_column_name)
    if label_parameter_dict[labels.CLASS_CUTOFFS_NAME] is None:
        label_column_names = [label_column_name]
    else:
        regression_label_column_name = (
            labels.get_column_name_for_regression_label(
                min_lead_time_sec=label_parameter_dict[
                    labels.MIN_LEAD_TIME_NAME],
                max_lead_time_sec=label_parameter_dict[
                    labels.MAX_LEAD_TIME_NAME],
                min_distance_metres=label_parameter_dict[
                    labels.MIN_DISTANCE_NAME],
                max_distance_metres=label_parameter_dict[
                    labels.MAX_DISTANCE_NAME],
                percentile_level=label_parameter_dict[
                    labels.PERCENTILE_LEVEL_NAME]))

        label_column_names = [label_column_name, regression_label_column_name]

    distance_buffer_column_names = tracking_io.get_distance_buffer_columns(
        storm_to_winds_table)
    if distance_buffer_column_names is None:
        distance_buffer_column_names = []

    storm_to_winds_table = storm_to_winds_table[
        STORM_TO_WIND_COLUMNS_TO_KEEP + label_column_names +
        distance_buffer_column_names]
    return feature_table.merge(
        storm_to_winds_table, on=COLUMNS_TO_MERGE_ON, how='inner')


def sample_by_min_observations(
        feature_table, min_observations=DEFAULT_MIN_OBSERVATIONS_FOR_SAMPLING,
        return_table=False):
    """Samples feature vectors, using the "min_observations" method.

    :param feature_table: pandas DataFrame created by
        join_features_and_label_for_storm_objects.
    :param min_observations: Minimum number of wind observations.  All storm
        objects with >= `min_observations` wind observations will be selected.
    :param return_table: Boolean flag.  If True, will return sampled feature
        vectors.  If False, will return metadata, which can be used to sample
        from a large number of feature tables in the future.
    :return: feature_table: If return_table = False, this is None.  If
        return_table = True, this is the input table with fewer rows.
    :return: metadata_dict: If return_table = True, this is None.  If
        return_table = False, dictionary with the following keys.
    metadata_dict['live_selected_indices']: 1-D numpy array with indices (rows)
        of selected feature vectors belonging to live storm objects.
    metadata_dict['live_indices']: 1-D numpy array with indices (rows) of all
        live storm objects.
    metadata_dict['dead_indices']: 1-D numpy array with indices (rows) of all
        dead storm objects.
    """

    error_checking.assert_is_integer(min_observations)
    error_checking.assert_is_greater(min_observations, 0)
    error_checking.assert_is_boolean(return_table)

    selected_flags = (
        feature_table[labels.NUM_OBSERVATIONS_FOR_LABEL_COLUMN].values >=
        min_observations)
    live_selected_indices = numpy.array(numpy.where(selected_flags)[0])
    live_indices, dead_indices = _find_live_and_dead_storms(feature_table)

    if not return_table:
        metadata_dict = {
            LIVE_SELECTED_INDICES_KEY: live_selected_indices,
            LIVE_INDICES_KEY: live_indices, DEAD_INDICES_KEY: dead_indices}
        return None, metadata_dict

    selected_indices = _select_dead_storms(
        live_indices=live_indices, dead_indices=dead_indices,
        live_selected_indices=live_selected_indices)
    return feature_table.iloc[selected_indices], None


def sample_by_min_observations_plus(
        feature_table, min_observations=DEFAULT_MIN_OBSERVATIONS_FOR_SAMPLING,
        return_table=False):
    """Samples feature vectors, using the "min_observations_plus" method.

    :param feature_table: pandas DataFrame created by
        join_features_and_label_for_storm_objects.
    :param min_observations: See documentation for sample_min_observations.
    :param return_table: See documentation for sample_min_observations.
    :return: feature_table: See documentation for sample_min_observations.
    :return: metadata_dict: See documentation for sample_min_observations.
    """

    error_checking.assert_is_integer(min_observations)
    error_checking.assert_is_greater(min_observations, 0)
    error_checking.assert_is_boolean(return_table)

    _, _, classification_label_column_name = check_feature_table(
        feature_table, require_storm_objects=True)
    label_parameter_dict = labels.column_name_to_label_params(
        classification_label_column_name)
    num_classes = 1 + len(label_parameter_dict[labels.CLASS_CUTOFFS_NAME])

    selected_flags = numpy.logical_or(
        feature_table[labels.NUM_OBSERVATIONS_FOR_LABEL_COLUMN].values >=
        min_observations,
        feature_table[classification_label_column_name].values ==
        num_classes - 1)

    live_selected_indices = numpy.array(numpy.where(selected_flags)[0])
    live_indices, dead_indices = _find_live_and_dead_storms(feature_table)

    if not return_table:
        metadata_dict = {
            LIVE_SELECTED_INDICES_KEY: live_selected_indices,
            LIVE_INDICES_KEY: live_indices, DEAD_INDICES_KEY: dead_indices}
        return None, metadata_dict

    selected_indices = _select_dead_storms(
        live_indices=live_indices, dead_indices=dead_indices,
        live_selected_indices=live_selected_indices)
    return feature_table.iloc[selected_indices], None


def sample_by_min_obs_density(
        feature_table,
        min_observation_density_m02=DEFAULT_MIN_OBS_DENSITY_FOR_SAMPLING_M02,
        return_table=False):
    """Samples feature vectors, using the "min_obs_density" method.

    :param feature_table: pandas DataFrame created by
        join_features_and_label_for_storm_objects.
    :param min_observation_density_m02: Minimum density of wind observations
        (number per m^2).  All storm objects with density >=
        `min_observation_density_m02` will be selected.
    :param return_table: See documentation for sample_by_min_observations.
    :return: feature_table: See documentation for sample_by_min_observations.
    :return: metadata_dict: See documentation for sample_by_min_observations.
    """

    error_checking.assert_is_greater(min_observation_density_m02, 0.)
    error_checking.assert_is_boolean(return_table)

    feature_table, observation_densities_m02 = _get_observation_densities(
        feature_table)
    selected_flags = observation_densities_m02 >= min_observation_density_m02
    live_selected_indices = numpy.array(numpy.where(selected_flags)[0])
    live_indices, dead_indices = _find_live_and_dead_storms(feature_table)

    if not return_table:
        metadata_dict = {
            LIVE_SELECTED_INDICES_KEY: live_selected_indices,
            LIVE_INDICES_KEY: live_indices, DEAD_INDICES_KEY: dead_indices}
        return None, metadata_dict

    selected_indices = _select_dead_storms(
        live_indices=live_indices, dead_indices=dead_indices,
        live_selected_indices=live_selected_indices)
    return feature_table.iloc[selected_indices], None


def sample_by_min_obs_density_plus(
        feature_table,
        min_observation_density_m02=DEFAULT_MIN_OBS_DENSITY_FOR_SAMPLING_M02,
        return_table=False):
    """Samples feature vectors, using the "min_obs_density_plus" method.

    :param feature_table: pandas DataFrame created by
        join_features_and_label_for_storm_objects.
    :param min_observation_density_m02: See documentation for
        sample_by_min_obs_density.
    :param return_table: See documentation for sample_by_min_observations.
    :return: feature_table: See documentation for sample_by_min_observations.
    :return: metadata_dict: See documentation for sample_by_min_observations.
    """

    error_checking.assert_is_greater(min_observation_density_m02, 0.)
    error_checking.assert_is_boolean(return_table)

    feature_table, observation_densities_m02 = _get_observation_densities(
        feature_table)

    _, _, classification_label_column_name = check_feature_table(
        feature_table, require_storm_objects=True)
    label_parameter_dict = labels.column_name_to_label_params(
        classification_label_column_name)
    num_classes = 1 + len(label_parameter_dict[labels.CLASS_CUTOFFS_NAME])

    selected_flags = numpy.logical_or(
        observation_densities_m02 >= min_observation_density_m02,
        feature_table[classification_label_column_name].values ==
        num_classes - 1)

    live_selected_indices = numpy.array(numpy.where(selected_flags)[0])
    live_indices, dead_indices = _find_live_and_dead_storms(feature_table)

    if not return_table:
        metadata_dict = {
            LIVE_SELECTED_INDICES_KEY: live_selected_indices,
            LIVE_INDICES_KEY: live_indices, DEAD_INDICES_KEY: dead_indices}
        return None, metadata_dict

    selected_indices = _select_dead_storms(
        live_indices=live_indices, dead_indices=dead_indices,
        live_selected_indices=live_selected_indices)
    return feature_table.iloc[selected_indices], None


def sample_by_uniform_wind_speed(
        feature_table, cutoffs_m_s01=DEFAULT_CUTOFFS_FOR_UNIFORM_SAMPLING_M_S01,
        return_table=False):
    """Samples feature vectors, using the "uniform_wind_speed" method.

    U_q = [q]th percentile of wind speeds linked to a given storm object.  For a
    more thorough definition, see the docstring at the top.

    N = number of storm objects

    :param feature_table: N-row pandas DataFrame created by
        join_features_and_label_for_storm_objects.
    :param cutoffs_m_s01: 1-D numpy array of cutoffs (metres per second), which
        will be used to create categories for U_q.  An equal number of storm
        objects in each U_q category will be selected.
    :param return_table: Boolean flag.  If True, will return sampled feature
        vectors.  If False, will return metadata, which can be used to sample
        from a large number of feature tables in the future.
    :return: feature_table: If return_table = False, this is None.  If
        return_table = True, this is the input table with fewer rows.
    :return: metadata_dict: If return_table = True, this is None.  If
        return_table = False, dictionary with the following keys.
    metadata_dict['speed_category_by_storm_object']: length-N numpy array of
        wind-speed categories.  Category -1 means a dead storm object.
    metadata_dict['num_observations_by_storm_object']: length-N numpy array,
        where the [i]th element is number of wind observations used to label the
        [i]th storm object.
    """

    _, regression_label_column_name, _ = check_feature_table(
        feature_table, require_storm_objects=True)
    speed_category_by_storm_object = classifn_utils.classify_values(
        feature_table[regression_label_column_name].values, cutoffs_m_s01,
        non_negative_only=True)

    _, dead_indices = _find_live_and_dead_storms(feature_table)
    speed_category_by_storm_object[dead_indices] = -1
    num_observations_by_storm_object = feature_table[
        labels.NUM_OBSERVATIONS_FOR_LABEL_COLUMN].values

    if not return_table:
        metadata_dict = {SPEED_CATEGORY_KEY_FOR_UNIFORM_SAMPLING:
                             speed_category_by_storm_object,
                         NUM_OBSERVATIONS_KEY: num_observations_by_storm_object}
        return None, metadata_dict

    selected_indices = _select_storms_uniformly_by_category(
        speed_category_by_storm_object, num_observations_by_storm_object)
    return feature_table.iloc[selected_indices], None


def sample_many_files_by_min_obs_or_density(
        input_feature_file_names, output_feature_file_names,
        sampling_method=None,
        min_observations=DEFAULT_MIN_OBSERVATIONS_FOR_SAMPLING,
        min_observation_density_m02=DEFAULT_MIN_OBS_DENSITY_FOR_SAMPLING_M02):
    """Samples feature vectors from many files.

    For any sampling method other than "uniform_wind_speed".  To sample from
    many files with the "uniform_wind_speed" method, see
    sample_many_files_by_uniform_wind_speed.

    N = number of input files

    :param input_feature_file_names: length-N list with paths to files
        containing unsampled feature vectors.
    :param output_feature_file_names: length-N list with paths to files
        containing sampled feature vectors.
    :param sampling_method: Sampling method.  Must be "min_observations",
        "min_observations_plus", "min_obs_density", or "min_obs_density_plus".
    :param min_observations: See documentation for sample_by_min_observations.
    :param min_observation_density_m02: See documentation for
        sample_by_min_obs_density.
    :raises: ValueError: All input files must have the same label columns.  In
        other words, labels must have been created with the same params for each
        file.
    :raises: ValueError: if sampling method is invalid.
    """

    error_checking.assert_is_string_list(input_feature_file_names)
    error_checking.assert_is_numpy_array(
        numpy.asarray(input_feature_file_names), num_dimensions=1)
    for this_file_name in input_feature_file_names:
        error_checking.assert_file_exists(this_file_name)
    num_files = len(input_feature_file_names)

    error_checking.assert_is_string_list(output_feature_file_names)
    error_checking.assert_is_numpy_array(
        numpy.asarray(output_feature_file_names),
        exact_dimensions=numpy.array([num_files]))

    error_checking.assert_is_string(sampling_method)
    if sampling_method not in NON_UNIFORM_SAMPLING_METHODS:
        error_string = (
            '\n\n' + str(NON_UNIFORM_SAMPLING_METHODS) + '\n\nValid sampling ' +
            'methods (other than "' + UNIFORM_SAMPLING_METHOD +
            '"; listed above) do not include "' + sampling_method + '".')
        raise ValueError(error_string)

    live_selected_indices_by_file = [None] * num_files
    live_indices_by_file = [None] * num_files
    dead_indices_by_file = [None] * num_files
    num_storm_objects_by_file = numpy.full(num_files, 0, dtype=int)

    for i in range(num_files):
        file_system_utils.mkdir_recursive_if_necessary(
            file_name=output_feature_file_names[i])

        print ('Sampling data from file ' + str(i + 1) + '/' + str(num_files) +
               ': ' + input_feature_file_names[i] + '...')

        this_feature_table = read_features_for_storm_objects(
            input_feature_file_names[i])

        if i == 0:
            _, regression_label_column_name, _ = check_feature_table(
                this_feature_table, require_storm_objects=True)
        else:
            _, this_regression_label_column_name, _ = check_feature_table(
                this_feature_table, require_storm_objects=True)
            if (this_regression_label_column_name !=
                    regression_label_column_name):
                error_string = (
                    'Files ' + str(i + 1) + ' and 1 have different regression-'
                    + 'label columns ("' + this_regression_label_column_name +
                    '" and "' + regression_label_column_name +
                    '", respectively).')
                raise ValueError(error_string)

        if sampling_method == MIN_OBS_SAMPLING_METHOD:
            _, this_metadata_dict = sample_by_min_observations(
                this_feature_table, min_observations=min_observations,
                return_table=False)
        elif sampling_method == MIN_OBS_PLUS_SAMPLING_METHOD:
            _, this_metadata_dict = sample_by_min_observations_plus(
                this_feature_table, min_observations=min_observations,
                return_table=False)
        elif sampling_method == MIN_DENSITY_SAMPLING_METHOD:
            _, this_metadata_dict = sample_by_min_obs_density(
                this_feature_table,
                min_observation_density_m02=min_observation_density_m02,
                return_table=False)
        elif sampling_method == MIN_DENSITY_PLUS_SAMPLING_METHOD:
            _, this_metadata_dict = sample_by_min_obs_density_plus(
                this_feature_table,
                min_observation_density_m02=min_observation_density_m02,
                return_table=False)

        live_selected_indices_by_file[i] = this_metadata_dict[
            LIVE_SELECTED_INDICES_KEY]
        live_indices_by_file[i] = this_metadata_dict[LIVE_INDICES_KEY]
        dead_indices_by_file[i] = this_metadata_dict[DEAD_INDICES_KEY]
        num_storm_objects_by_file[i] = len(this_feature_table.index)

    live_selected_indices = _indices_from_file_specific_to_overall(
        live_selected_indices_by_file, num_storm_objects_by_file)
    live_indices = _indices_from_file_specific_to_overall(
        live_indices_by_file, num_storm_objects_by_file)
    dead_indices = _indices_from_file_specific_to_overall(
        dead_indices_by_file, num_storm_objects_by_file)

    selected_indices = _select_dead_storms(
        live_indices=live_indices, dead_indices=dead_indices,
        live_selected_indices=live_selected_indices)
    selected_indices_by_file = _indices_from_overall_to_file_specific(
        selected_indices, num_storm_objects_by_file)

    for i in range(num_files):
        print ('Writing sampled data to file ' + str(i - 1) + '/' +
               str(num_files) + ': ' + output_feature_file_names[i] + '...')

        this_feature_table = read_features_for_storm_objects(
            input_feature_file_names[i])
        write_features_for_storm_objects(
            this_feature_table.iloc[selected_indices_by_file[i]],
            output_feature_file_names[i])


def sample_many_files_by_uniform_wind_speed(
        input_feature_file_names, output_feature_file_names,
        cutoffs_m_s01=DEFAULT_CUTOFFS_FOR_UNIFORM_SAMPLING_M_S01):
    """Samples feature vectors from many files by "uniform_wind_speed" method.

    N = number of input files

    :param input_feature_file_names: length-N list with paths to files
        containing unsampled feature vectors.
    :param output_feature_file_names: length-N list with paths to files
        containing sampled feature vectors.
    :param cutoffs_m_s01: See documentation for sample_by_uniform_wind_speed.
    :raises: ValueError: All input files must have the same label columns.  In
        other words, labels must have been created with the same params for each
        file.
    """

    error_checking.assert_is_string_list(input_feature_file_names)
    error_checking.assert_is_numpy_array(
        numpy.asarray(input_feature_file_names), num_dimensions=1)
    for this_file_name in input_feature_file_names:
        error_checking.assert_file_exists(this_file_name)
    num_files = len(input_feature_file_names)

    error_checking.assert_is_string_list(output_feature_file_names)
    error_checking.assert_is_numpy_array(
        numpy.asarray(output_feature_file_names),
        exact_dimensions=numpy.array([num_files]))

    speed_category_by_storm_object = numpy.array([], dtype=int)
    num_observations_by_storm_object = numpy.array([], dtype=int)
    num_storm_objects_by_file = numpy.full(num_files, 0, dtype=int)

    for i in range(num_files):
        file_system_utils.mkdir_recursive_if_necessary(
            file_name=output_feature_file_names[i])

        print ('Sampling data from file ' + str(i - 1) + '/' + str(num_files) +
               ': ' + input_feature_file_names[i] + '...')

        this_feature_table = read_features_for_storm_objects(
            input_feature_file_names[i])

        if i == 0:
            _, regression_label_column_name, _ = check_feature_table(
                this_feature_table, require_storm_objects=True)
        else:
            _, this_regression_label_column_name, _ = check_feature_table(
                this_feature_table, require_storm_objects=True)
            if (this_regression_label_column_name !=
                    regression_label_column_name):
                error_string = (
                    'Files ' + str(i + 1) + ' and 1 have different regression-'
                    + 'label columns ("' + this_regression_label_column_name +
                    '" and "' + regression_label_column_name +
                    '", respectively).')
                raise ValueError(error_string)

        _, this_metadata_dict = sample_by_uniform_wind_speed(
            this_feature_table, cutoffs_m_s01=cutoffs_m_s01, return_table=False)

        speed_category_by_storm_object = numpy.concatenate((
            speed_category_by_storm_object,
            this_metadata_dict[SPEED_CATEGORY_KEY_FOR_UNIFORM_SAMPLING]))
        num_observations_by_storm_object = numpy.concatenate((
            num_observations_by_storm_object,
            this_metadata_dict[NUM_OBSERVATIONS_KEY]))
        num_storm_objects_by_file[i] = len(this_feature_table.index)

    selected_indices = _select_storms_uniformly_by_category(
        speed_category_by_storm_object, num_observations_by_storm_object)
    selected_indices_by_file = _indices_from_overall_to_file_specific(
        selected_indices, num_storm_objects_by_file)

    for i in range(num_files):
        print ('Writing sampled data to file ' + str(i - 1) + '/' +
               str(num_files) + ': ' + output_feature_file_names[i] + '...')

        this_feature_table = read_features_for_storm_objects(
            input_feature_file_names[i])
        write_features_for_storm_objects(
            this_feature_table.iloc[selected_indices_by_file[i]],
            output_feature_file_names[i])


def write_features_for_storm_objects(feature_table, pickle_file_name):
    """Writes features for storm objects to a Pickle file.

    :param feature_table: pandas DataFrame created by
        join_features_and_label_for_storm_objects.
    :param pickle_file_name: Path to output file.
    """

    (feature_column_names,
     regression_label_column_name,
     classification_label_column_name) = check_feature_table(
         feature_table, require_storm_objects=True)
    distance_buffer_column_names = tracking_io.get_distance_buffer_columns(
        feature_table)
    columns_to_write = (
        STORM_TO_WIND_COLUMNS_TO_KEEP + feature_column_names +
        distance_buffer_column_names)

    if regression_label_column_name is not None:
        columns_to_write += [regression_label_column_name]
    if classification_label_column_name is not None:
        columns_to_write += [classification_label_column_name]

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(feature_table[columns_to_write], pickle_file_handle)
    pickle_file_handle.close()


def read_features_for_storm_objects(pickle_file_name):
    """Reads features for storm objects from a Pickle file.

    :param pickle_file_name: Path to input file.
    :return: feature_table: pandas DataFrame with columns documented in
        join_features_and_label_for_storm_objects.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    feature_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    check_feature_table(feature_table, require_storm_objects=True)
    return feature_table


def find_unsampled_file_one_time(
        unix_time_sec=None, spc_date_unix_sec=None, top_directory_name=None,
        raise_error_if_missing=True):
    """Locates file with unsampled feature vectors for one time step.

    :param unix_time_sec: Time step (valid time).
    :param spc_date_unix_sec: SPC (Storm Prediction Center) date.
    :param top_directory_name: Name of top-level directory with feature files.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will raise an error.
    :return: unsampled_file_name: Path to file with unsampled feature vectors
        for one time step.  If raise_error_if_missing = False and file is
        missing, this will be the *expected* path.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    pathless_file_name = '{0:s}_{1:s}{2:s}'.format(
        FEATURE_FILE_PREFIX,
        time_conversion.unix_sec_to_string(
            unix_time_sec, TIME_FORMAT_IN_FILE_NAMES), FEATURE_FILE_EXTENSION)

    unsampled_file_name = '{0:s}/{1:s}/{2:s}'.format(
        top_directory_name,
        time_conversion.time_to_spc_date_string(spc_date_unix_sec),
        pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(unsampled_file_name):
        raise ValueError(
            'Cannot find file with feature vectors.  Expected at location: ' +
            unsampled_file_name)

    return unsampled_file_name


def find_unsampled_file_time_period(
        start_time_unix_sec, end_time_unix_sec, directory_name=None,
        raise_error_if_missing=True):
    """Locates file with unsampled feature vectors for time period.

    :param start_time_unix_sec: Beginning of time period.
    :param end_time_unix_sec: End of time period.
    :param directory_name: Name of directory.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will raise an error.
    :return: unsampled_file_name: Path to file with unsampled feature vectors
        for time period.  If raise_error_if_missing = False and file is missing,
        this will be the *expected* path.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    start_time_string = time_conversion.unix_sec_to_string(
        start_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES)
    end_time_string = time_conversion.unix_sec_to_string(
        end_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES)
    error_checking.assert_is_geq(end_time_unix_sec, start_time_unix_sec)

    pathless_file_name = '{0:s}_{1:s}_{2:s}{3:s}'.format(
        FEATURE_FILE_PREFIX, start_time_string, end_time_string,
        FEATURE_FILE_EXTENSION)
    unsampled_file_name = '{0:s}/{1:s}'.format(
        directory_name, pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(unsampled_file_name):
        raise ValueError(
            'Cannot find file with feature vectors.  Expected at location: ' +
            unsampled_file_name)

    return unsampled_file_name
