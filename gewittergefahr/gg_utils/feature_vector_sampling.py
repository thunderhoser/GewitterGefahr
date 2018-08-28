"""Methods for sampling feature vectors.

All sampling methods in this file are based on wind speed.  Thus, the label must
be based on wind speed.  In other words, the label must be one created by
`labels.label_wind_speed_for_classification` or
`labels.label_wind_speed_for_regression` -- as opposed to, for example,
`labels.label_tornado_occurrence`.  In due time I may add sampling methods based
on tornado occurrence.

--- GENERAL DEFINITIONS ---

Label = "dependent variable" = "target variable" = "outcome" = "predictand".
This is the variable that machine learning is trying to predict.  Examples are
max wind speed and tornado occurrence.

Feature = "attribute" = "predictor variable" = "independent variable" = variable
used to predict the label.

Feature vector = list of all features for one example (storm object).

--- DEFINITIONS FOR SAMPLING METHODS ---

q = wind-speed percentile level used to create either regression or
    classification label.  For more on the role of q, see documentation for
    `labels.label_wind_speed_for_classification` and
    `labels.label_wind_speed_for_regression`.

U_q = [q]th-percentile wind speed for all observations linked to a given storm
      object.

N_obs = number of wind observations used to create the label for a given storm
        object.

T_lead = range of lead times used to create label.

Live storm object = storm object for which the corresponding cell still exists
                    at some point during T_lead.

Dead storm object = storm object for which the corresponding cell does not exist
                    at any point during T_lead.

--- SAMPLING METHODS ---

[1] "uniform_wind_speed": Storm objects are sampled uniformly with respect to
    U_q.
[2] "min_observations": All storm objects with N_obs >= threshold are used.
    Also, dead storm objects are sampled randomly to preserve the ratio of live
    to dead storm objects.  This prevents survivor bias.
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

import numpy
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import link_events_to_storms as events2storms
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import feature_vectors
from gewittergefahr.gg_utils import classification_utils as classifn_utils
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6
KT_TO_METRES_PER_SECOND = 1.852 / 3.6

LIVE_SELECTED_INDICES_KEY = 'live_selected_indices'
LIVE_INDICES_KEY = 'live_indices'
DEAD_INDICES_KEY = 'dead_indices'
WIND_SPEED_CATEGORIES_KEY = 'wind_speed_categories'
WIND_OBSERVATION_COUNTS_KEY = 'wind_observation_counts'

UNIFORM_WIND_SPEED_METHOD = 'uniform_wind_speed'
MIN_OBSERVATION_COUNT_METHOD = 'min_observations'
MIN_OBS_COUNT_PLUS_SAMPLING_METHOD = 'min_observations_plus'
MIN_OBSERVATION_DENSITY_METHOD = 'min_obs_density'
MIN_OBS_DENSITY_PLUS_SAMPLING_METHOD = 'min_obs_density_plus'

DEFAULT_MIN_OBSERVATION_COUNT = 25
DEFAULT_MIN_OBSERVATION_DENSITY_M02 = 1e-7
DEFAULT_WIND_SPEED_CUTOFFS_FOR_SAMPLING_M_S01 = (
    KT_TO_METRES_PER_SECOND * numpy.array([10., 20., 30., 40., 50.]))

VALID_SAMPLING_METHODS = [
    UNIFORM_WIND_SPEED_METHOD, MIN_OBSERVATION_COUNT_METHOD,
    MIN_OBS_COUNT_PLUS_SAMPLING_METHOD, MIN_OBSERVATION_DENSITY_METHOD,
    MIN_OBS_DENSITY_PLUS_SAMPLING_METHOD]
NON_UNIFORM_WIND_SPEED_METHODS = [
    MIN_OBSERVATION_COUNT_METHOD, MIN_OBS_COUNT_PLUS_SAMPLING_METHOD,
    MIN_OBSERVATION_DENSITY_METHOD, MIN_OBS_DENSITY_PLUS_SAMPLING_METHOD]

COLUMNS_TO_MERGE_ON = [
    tracking_utils.STORM_ID_COLUMN, tracking_utils.TIME_COLUMN]
INPUT_COLUMNS_FOR_BUFFER_CREATION = COLUMNS_TO_MERGE_ON + [
    tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN]


def _find_live_and_dead_storms(
        storm_object_times_unix_sec, storm_cell_end_times_unix_sec,
        min_lead_time_sec):
    """Finds both live and dead storm objects.

    N = number of storm objects

    :param storm_object_times_unix_sec: length-N numpy array of storm-object
        times (valid times).
    :param storm_cell_end_times_unix_sec: length-N numpy array with end times of
        corresponding storm cells.
    :param min_lead_time_sec: Minimum lead time.  Any storm object with
        remaining lifetime >= `min_lead_time_sec` is considered "live"; all
        others are considered "dead".
    :return: live_indices: 1-D numpy array with indices (into
        `storm_object_times_unix_sec`) of live storm objects.
    :return: dead_indices: 1-D numpy array with indices (into
        `storm_object_times_unix_sec`) of dead storm objects.
    """

    remaining_storm_lifetimes_sec = (
        storm_cell_end_times_unix_sec - storm_object_times_unix_sec)
    live_flags = remaining_storm_lifetimes_sec >= min_lead_time_sec

    return (numpy.array(numpy.where(live_flags)[0]),
            numpy.array(numpy.where(numpy.invert(live_flags))[0]))


def _select_dead_storms_randomly(
        live_indices, dead_indices, live_selected_indices):
    """Randomly selects dead storm objects.

    See docstring for the definitions of "live" and "dead" storm objects.

    f_dead = fraction of storm objects that are dead

    This method selects dead storm objects so that f_dead in the selected
    dataset = f_dead in the full dataset.

    :param live_indices: 1-D numpy array with indices of live storm objects.
    :param dead_indices: 1-D numpy array with indices of dead storm objects.
    :param live_selected_indices: 1-D numpy array with indices of live storm
        objects that have been selected.
    :return: selected_indices: 1-D numpy array with indices of all storm objects
        (alive or dead) that have been selected.
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


def _select_storms_uniformly_by_category(
        storm_object_categories, storm_object_priorities):
    """Selects an equal number of storm objects from each category.

    Within each category, this method selects the storm objects with the highest
    priorities.

    N = number of storm objects

    :param storm_object_categories: length-N numpy array with integer category
        for each storm object.
    :param storm_object_priorities: length-N numpy array with priority for each
        storm object.
    :return: selected_indices: 1-D numpy array with indices of storm objects
        that have been selected.
    """

    unique_categories, orig_to_unique_category_indices = numpy.unique(
        storm_object_categories, return_inverse=True)

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
            -storm_object_priorities[these_storm_indices])

        these_selected_indices = (
            these_sort_indices[:min_storm_objects_in_category])
        these_selected_indices = these_storm_indices[these_selected_indices]
        selected_indices = numpy.concatenate((
            selected_indices, these_selected_indices))

    return numpy.sort(selected_indices)


def _get_wind_observation_densities(
        feature_table, min_lead_time_sec, max_lead_time_sec,
        min_link_distance_metres, max_link_distance_metres):
    """Finds wind-observation density* around each storm object.

    * Within the given lead-time and distance windows.

    t_w = time of wind observation
    t_s = time of storm object
    "Lead time" = t_w - t_s
    N = number of storm objects

    :param feature_table: N-row pandas DataFrame with columns listed in
        `feature_vectors.write_feature_table`.
    :param min_lead_time_sec: Minimum lead time.
    :param max_lead_time_sec: Max lead time.
    :param min_link_distance_metres: Minimum linkage distance (between storm
        boundary and wind observation).
    :param max_link_distance_metres: Max linkage distance.
    :return: feature_table: Same as input, but maybe with additional column
        containing distance buffers.
    :return: observation_densities_m02: length-N numpy array of wind-observation
        densities (number per m^2).
    """

    if min_link_distance_metres < TOLERANCE:
        min_link_distance_metres = numpy.nan
    buffer_column_name = tracking_utils.distance_buffer_to_column_name(
        min_link_distance_metres, max_link_distance_metres)

    if buffer_column_name not in feature_table:
        buffer_table = tracking_utils.make_buffers_around_storm_objects(
            feature_table[INPUT_COLUMNS_FOR_BUFFER_CREATION],
            min_distances_metres=numpy.array([min_link_distance_metres]),
            max_distances_metres=numpy.array([max_link_distance_metres]))

        feature_table = feature_table.merge(
            buffer_table, on=COLUMNS_TO_MERGE_ON, how='inner')

    num_storm_objects = len(feature_table.index)
    buffer_areas_m2 = numpy.full(num_storm_objects, numpy.nan)
    for i in range(num_storm_objects):
        this_polygon_object_xy, _ = polygons.project_latlng_to_xy(
            feature_table[buffer_column_name].values[i])
        buffer_areas_m2[i] = this_polygon_object_xy.area

    num_wind_obs_column_name = labels.get_column_name_for_num_wind_obs(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres)

    observation_densities_m02 = (
        feature_table[num_wind_obs_column_name].values.astype(float) /
        buffer_areas_m2)
    return feature_table, observation_densities_m02


def _indices_from_file_specific_to_overall(
        indices_by_file, num_objects_by_file):
    """Converts array indices from file-specific to overall.

    For example, in a one-based indexing system: if file 1 contains 50 objects
    and file 2 contains 100 objects, object 3 in file 3 will be object 153
    overall (50 + 100 + 3).

    In a zero-based indexing system: if file 1 contains 50 objects
    and file 2 contains 100 objects, object 3 in file 3 will also be object 153
    overall (50 + 100 + [3 + 1] - 1).

    N = number of files

    :param indices_by_file: length-N list, where each element is a 1-D numpy
        array of object indices.
    :param num_objects_by_file: length-N numpy array with number of objects in
        each file.
    :return: overall_indices: 1-D numpy array of overall object indices.
    """

    # TODO(thunderhoser): This method should be somewhere else.

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


def _indices_from_overall_to_file_specific(
        overall_indices, num_objects_by_file):
    """Converts array indices from overall to file-specific.

    This method is the inverse of `_indices_from_file_specific_to_overall`.

    N = number of files

    :param overall_indices: 1-D numpy array of overall object indices.
    :param num_objects_by_file: length-N numpy array with number of objects in
        each file.
    :return: indices_by_file: length-N list, where each element is a 1-D numpy
        array of object indices.
    """

    # TODO(thunderhoser): This method should be somewhere else.

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


def sample_by_min_observations(
        feature_table, min_lead_time_sec, max_lead_time_sec,
        min_link_distance_metres, max_link_distance_metres,
        min_observation_count=DEFAULT_MIN_OBSERVATION_COUNT,
        return_table=False):
    """Samples storm objects via the "min_observations" method.

    Briefly, this method keeps all storm objects with >= `min_observation_count`
    wind observations in the given lead-time and distance windows.

    t_w = time of wind observation
    t_s = time of storm object
    "Lead time" = t_w - t_s

    :param feature_table: pandas DataFrame with columns listed in
        `feature_vectors.write_feature_table`.
    :param min_lead_time_sec: Minimum lead time.
    :param max_lead_time_sec: Max lead time.
    :param min_link_distance_metres: Minimum linkage distance (between storm
        boundary and wind observation).
    :param max_link_distance_metres: Max linkage distance.
    :param min_observation_count: Threshold on number of wind observations
        linked to storm object.
    :param return_table: Boolean flag.  Determines which variable is returned.
    :return: feature_table: [None if return_table = False] Same as the input
        table, but with fewer rows.
    :return: metadata_dict: [None if return_table = True] Dictionary with the
        following keys.
    metadata_dict['live_selected_indices']: 1-D numpy array with indices (rows
        in `feature_table`) of live storm objects that have been selected.
    metadata_dict['live_indices']: 1-D numpy array with indices (rows in
        `feature_table`) of live storm objects.
    metadata_dict['dead_indices']: 1-D numpy array with indices (rows in
        `feature_table`) of dead storm objects.
    """

    num_wind_obs_column_name = labels.get_column_name_for_num_wind_obs(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres)

    error_checking.assert_is_integer(min_observation_count)
    error_checking.assert_is_greater(min_observation_count, 0)
    error_checking.assert_is_boolean(return_table)

    selected_flags = (
        feature_table[num_wind_obs_column_name].values >= min_observation_count)

    live_selected_indices = numpy.array(numpy.where(selected_flags)[0])
    live_indices, dead_indices = _find_live_and_dead_storms(
        storm_object_times_unix_sec=feature_table[tracking_utils.TIME_COLUMN],
        storm_cell_end_times_unix_sec=feature_table[
            tracking_utils.CELL_END_TIME_COLUMN].values,
        min_lead_time_sec=min_lead_time_sec)

    if not return_table:
        metadata_dict = {
            LIVE_SELECTED_INDICES_KEY: live_selected_indices,
            LIVE_INDICES_KEY: live_indices, DEAD_INDICES_KEY: dead_indices
        }
        return None, metadata_dict

    selected_indices = _select_dead_storms_randomly(
        live_indices=live_indices, dead_indices=dead_indices,
        live_selected_indices=live_selected_indices)
    return feature_table.iloc[selected_indices], None


def sample_by_min_observations_plus(
        feature_table, min_lead_time_sec, max_lead_time_sec,
        min_link_distance_metres, max_link_distance_metres,
        wind_speed_percentile_level, wind_speed_class_cutoffs_kt,
        min_observation_count=DEFAULT_MIN_OBSERVATION_COUNT,
        return_table=False):
    """Samples storm objects via the "min_observations_plus" method.

    Briefly, this method keeps all storm objects meeting one of two criteria:

    [1] Number of wind obs linked to storm object, in the given lead-time and
        distance windows, is >= `min_observation_count`.
    [2] Wind-speed-classification label is the highest (most severe) possible.

    :param feature_table: See documentation for `sample_by_min_observations`.
    :param min_lead_time_sec: See doc for `sample_by_min_observations`.
    :param max_lead_time_sec: See doc for `sample_by_min_observations`.
    :param min_link_distance_metres: See doc for `sample_by_min_observations`.
    :param max_link_distance_metres: See doc for `sample_by_min_observations`.
    :param wind_speed_percentile_level: Percentile level used to create label.
    :param wind_speed_class_cutoffs_kt: Class cutoffs used to create label (in
        knots, or nautical miles per hour).
    :param min_observation_count: See doc for `sample_by_min_observations`.
    :param return_table: See doc for `sample_by_min_observations`.
    :return: feature_table: See doc for `sample_by_min_observations`.
    :return: metadata_dict: See doc for `sample_by_min_observations`.
    """

    label_column_name = labels.get_column_name_for_classification_label(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres,
        event_type_string=events2storms.WIND_EVENT_TYPE_STRING,
        wind_speed_percentile_level=wind_speed_percentile_level,
        wind_speed_class_cutoffs_kt=wind_speed_class_cutoffs_kt)

    num_wind_obs_column_name = labels.get_column_name_for_num_wind_obs(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres)

    error_checking.assert_is_integer(min_observation_count)
    error_checking.assert_is_greater(min_observation_count, 0)
    error_checking.assert_is_boolean(return_table)

    label_parameter_dict = labels.column_name_to_label_params(label_column_name)
    num_classes = 1 + len(
        label_parameter_dict[labels.WIND_SPEED_CLASS_CUTOFFS_KEY])

    selected_flags = numpy.logical_or(
        feature_table[num_wind_obs_column_name].values >= min_observation_count,
        feature_table[label_column_name].values == num_classes - 1)

    live_selected_indices = numpy.array(numpy.where(selected_flags)[0])
    live_indices, dead_indices = _find_live_and_dead_storms(
        storm_object_times_unix_sec=feature_table[tracking_utils.TIME_COLUMN],
        storm_cell_end_times_unix_sec=feature_table[
            tracking_utils.CELL_END_TIME_COLUMN].values,
        min_lead_time_sec=min_lead_time_sec)

    if not return_table:
        metadata_dict = {
            LIVE_SELECTED_INDICES_KEY: live_selected_indices,
            LIVE_INDICES_KEY: live_indices, DEAD_INDICES_KEY: dead_indices
        }
        return None, metadata_dict

    selected_indices = _select_dead_storms_randomly(
        live_indices=live_indices, dead_indices=dead_indices,
        live_selected_indices=live_selected_indices)
    return feature_table.iloc[selected_indices], None


def sample_by_min_obs_density(
        feature_table, min_lead_time_sec, max_lead_time_sec,
        min_link_distance_metres, max_link_distance_metres,
        min_observation_density_m02=DEFAULT_MIN_OBSERVATION_DENSITY_M02,
        return_table=False):
    """Samples storm objects via the "min_obs_density" method.

    Briefly, this method keeps all storm objects with wind-observation density
    >= `min_observation_density_m02` in the given lead-time and distance
    windows.

    :param feature_table: See documentation for `sample_by_min_observations`.
    :param min_lead_time_sec: See doc for `sample_by_min_observations`.
    :param max_lead_time_sec: See doc for `sample_by_min_observations`.
    :param min_link_distance_metres: See doc for `sample_by_min_observations`.
    :param max_link_distance_metres: See doc for `sample_by_min_observations`.
    :param min_observation_density_m02: Threshold on spatial density of wind
        observations linked to storm object (number per m^2).
    :param return_table: See doc for `sample_by_min_observations`.
    :return: feature_table: See doc for `sample_by_min_observations`.
    :return: metadata_dict: See doc for `sample_by_min_observations`.
    """

    error_checking.assert_is_greater(min_observation_density_m02, 0.)
    error_checking.assert_is_boolean(return_table)

    feature_table, observation_densities_m02 = _get_wind_observation_densities(
        feature_table=feature_table, min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres)

    selected_flags = observation_densities_m02 >= min_observation_density_m02

    live_selected_indices = numpy.array(numpy.where(selected_flags)[0])
    live_indices, dead_indices = _find_live_and_dead_storms(
        storm_object_times_unix_sec=feature_table[tracking_utils.TIME_COLUMN],
        storm_cell_end_times_unix_sec=feature_table[
            tracking_utils.CELL_END_TIME_COLUMN].values,
        min_lead_time_sec=min_lead_time_sec)

    if not return_table:
        metadata_dict = {
            LIVE_SELECTED_INDICES_KEY: live_selected_indices,
            LIVE_INDICES_KEY: live_indices, DEAD_INDICES_KEY: dead_indices
        }
        return None, metadata_dict

    selected_indices = _select_dead_storms_randomly(
        live_indices=live_indices, dead_indices=dead_indices,
        live_selected_indices=live_selected_indices)
    return feature_table.iloc[selected_indices], None


def sample_by_min_obs_density_plus(
        feature_table, min_lead_time_sec, max_lead_time_sec,
        min_link_distance_metres, max_link_distance_metres,
        wind_speed_percentile_level, wind_speed_class_cutoffs_kt,
        min_observation_density_m02=DEFAULT_MIN_OBSERVATION_DENSITY_M02,
        return_table=False):
    """Samples storm objects via the "min_obs_density_plus" method.

    Briefly, this method keeps all storm objects meeting one of two criteria:

    [1] Spatial density of wind obs linked to storm object, in the given
        lead-time and distance windows, is >= `min_observation_density_m02`.
    [2] Wind-speed-classification label is the highest (most severe) possible.

    :param feature_table: See documentation for
        `sample_by_min_observations_plus`.
    :param min_lead_time_sec: See doc for `sample_by_min_observations_plus`.
    :param max_lead_time_sec: See doc for `sample_by_min_observations_plus`.
    :param min_link_distance_metres: See doc for
        `sample_by_min_observations_plus`.
    :param max_link_distance_metres: See doc for
        `sample_by_min_observations_plus`.
    :param wind_speed_percentile_level: See doc for
        `sample_by_min_observations_plus`.
    :param wind_speed_class_cutoffs_kt: See doc for
        `sample_by_min_observations_plus`.
    :param min_observation_density_m02: Threshold on spatial density of wind
        observations linked to storm object (number per m^2).
    :param return_table: See doc for `sample_by_min_observations_plus`.
    :return: feature_table: See doc for `sample_by_min_observations_plus`.
    :return: metadata_dict: See doc for `sample_by_min_observations_plus`.
    """

    feature_table, observation_densities_m02 = _get_wind_observation_densities(
        feature_table=feature_table, min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres)

    label_column_name = labels.get_column_name_for_classification_label(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres,
        event_type_string=events2storms.WIND_EVENT_TYPE_STRING,
        wind_speed_percentile_level=wind_speed_percentile_level,
        wind_speed_class_cutoffs_kt=wind_speed_class_cutoffs_kt)

    error_checking.assert_is_greater(min_observation_density_m02, 0.)
    error_checking.assert_is_boolean(return_table)

    label_parameter_dict = labels.column_name_to_label_params(label_column_name)
    num_classes = 1 + len(
        label_parameter_dict[labels.WIND_SPEED_CLASS_CUTOFFS_KEY])

    selected_flags = numpy.logical_or(
        observation_densities_m02 >= min_observation_density_m02,
        feature_table[label_column_name].values == num_classes - 1)

    live_selected_indices = numpy.array(numpy.where(selected_flags)[0])
    live_indices, dead_indices = _find_live_and_dead_storms(
        storm_object_times_unix_sec=feature_table[tracking_utils.TIME_COLUMN],
        storm_cell_end_times_unix_sec=feature_table[
            tracking_utils.CELL_END_TIME_COLUMN].values,
        min_lead_time_sec=min_lead_time_sec)

    if not return_table:
        metadata_dict = {
            LIVE_SELECTED_INDICES_KEY: live_selected_indices,
            LIVE_INDICES_KEY: live_indices, DEAD_INDICES_KEY: dead_indices
        }
        return None, metadata_dict

    selected_indices = _select_dead_storms_randomly(
        live_indices=live_indices, dead_indices=dead_indices,
        live_selected_indices=live_selected_indices)
    return feature_table.iloc[selected_indices], None


def sample_by_uniform_wind_speed(
        feature_table, min_lead_time_sec, max_lead_time_sec,
        min_link_distance_metres, max_link_distance_metres,
        wind_speed_percentile_level,
        sampling_cutoffs_m_s01=DEFAULT_WIND_SPEED_CUTOFFS_FOR_SAMPLING_M_S01,
        return_table=False):
    """Samples storm objects via the "uniform_wind_speed" method.

    Briefly, this method selects an equal number of storm objects from each
    wind-speed category defined by `wind_speed_percentile_level` and
    `sampling_cutoffs_m_s01`.

    t_w = time of wind observation
    t_s = time of storm object
    "Lead time" = t_w - t_s

    q = percentile level for wind speed
    U_q = [q]th-percentile wind speed for a given storm object

    :param feature_table: See documentation for `sample_by_min_observations`.
    :param min_lead_time_sec: Minimum lead time.
    :param max_lead_time_sec: Max lead time.
    :param min_link_distance_metres: Minimum linkage distance (between storm
        boundary and wind observation).
    :param max_link_distance_metres: Max linkage distance.
    :param wind_speed_percentile_level: Percentile level for wind speed.
    :param sampling_cutoffs_m_s01: 1-D numpy array of cutoffs used to define
        categories for U_q.
    :param return_table: Boolean flag.  Determines which variable is returned.
    :return: feature_table: [None if return_table = False] Same as the input
        table, but with fewer rows.
    :return: metadata_dict: [None if return_table = True] Dictionary with the
        following keys.
    metadata_dict['wind_speed_categories']: length-N numpy array of
        wind-speed categories (integers), where -1 means a dead storm object.
    metadata_dict['wind_observation_counts']: length-N numpy array of wind-
        observation counts (number of wind observations used to create label for
        each storm object).
    """

    num_wind_obs_column_name = labels.get_column_name_for_num_wind_obs(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres)

    label_column_name = labels.get_column_name_for_regression_label(
        min_lead_time_sec=min_lead_time_sec,
        max_lead_time_sec=max_lead_time_sec,
        min_link_distance_metres=min_link_distance_metres,
        max_link_distance_metres=max_link_distance_metres,
        wind_speed_percentile_level=wind_speed_percentile_level)

    wind_speed_categories = classifn_utils.classify_values(
        input_values=feature_table[label_column_name].values,
        class_cutoffs=sampling_cutoffs_m_s01, non_negative_only=True)

    _, dead_indices = _find_live_and_dead_storms(
        storm_object_times_unix_sec=feature_table[tracking_utils.TIME_COLUMN],
        storm_cell_end_times_unix_sec=feature_table[
            tracking_utils.CELL_END_TIME_COLUMN].values,
        min_lead_time_sec=min_lead_time_sec)

    wind_speed_categories[dead_indices] = -1
    wind_observation_counts = feature_table[num_wind_obs_column_name].values

    if not return_table:
        metadata_dict = {
            WIND_SPEED_CATEGORIES_KEY: wind_speed_categories,
            WIND_OBSERVATION_COUNTS_KEY: wind_observation_counts
        }
        return None, metadata_dict

    selected_indices = _select_storms_uniformly_by_category(
        storm_object_categories=wind_speed_categories,
        storm_object_priorities=wind_observation_counts)
    return feature_table.iloc[selected_indices], None


def sample_by_min_obs_or_density_many_files(
        input_feature_file_names, output_feature_file_names, sampling_method,
        min_lead_time_sec, max_lead_time_sec,
        min_link_distance_metres, max_link_distance_metres,
        wind_speed_percentile_level=None, wind_speed_class_cutoffs_kt=None,
        min_observation_count=DEFAULT_MIN_OBSERVATION_COUNT,
        min_observation_density_m02=DEFAULT_MIN_OBSERVATION_DENSITY_M02):
    """Samples storm objects from many files.

    May use one of the following sampling methods, all of which are defined in
    the docstring:

    - "min_observations"
    - "min_observations_plus"
    - "min_obs_density"
    - "min_obs_density_plus"

    N = number of input files

    :param input_feature_file_names: length-N list of paths to input files (with
        unsampled feature vectors).
    :param output_feature_file_names: length-N list of paths to output files
        (for sampled feature vectors).
    :param sampling_method: Sampling method (one of the 4 strings listed above).
    :param min_lead_time_sec: See documentation for
        `sample_by_min_observations`.  This option is used by all 4 sampling
        methods.
    :param max_lead_time_sec: See doc for `sample_by_min_observations`.  This
        option is used by all 4 sampling methods.
    :param min_link_distance_metres: See doc for `sample_by_min_observations`.
        This option is used by all 4 sampling methods.
    :param max_link_distance_metres: See doc for `sample_by_min_observations`.
        This option is used by all 4 sampling methods.
    :param wind_speed_percentile_level: See doc for
        `sample_by_min_observations_plus`.  This option is used only by
        "min_observations_plus" and "min_obs_density_plus".
    :param wind_speed_class_cutoffs_kt: See doc for
        `sample_by_min_observations_plus`.  This option is used only by
        "min_observations_plus" and "min_obs_density_plus".
    :param min_observation_count: See doc for `sample_by_min_observations`.
        This option is used only by "min_observations" and
        "min_observations_plus".
    :param min_observation_density_m02: See doc for `sample_by_min_obs_density`.
        This option is used only by "min_obs_density" and
        "min_obs_density_plus".
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
    if sampling_method not in NON_UNIFORM_WIND_SPEED_METHODS:
        error_string = (
            '\n\n{0:s}\n\nValid sampling methods (listed above) do not include '
            '"{1:s}"').format(NON_UNIFORM_WIND_SPEED_METHODS, sampling_method)
        raise ValueError(error_string)

    live_selected_indices_by_file = [None] * num_files
    live_indices_by_file = [None] * num_files
    dead_indices_by_file = [None] * num_files
    num_storm_objects_by_file = numpy.full(num_files, -1, dtype=int)

    for i in range(num_files):
        file_system_utils.mkdir_recursive_if_necessary(
            file_name=output_feature_file_names[i])

        print 'Reading unsampled feature vectors from: "{0:s}"...'.format(
            input_feature_file_names[i])
        this_feature_table = feature_vectors.read_feature_table(
            input_feature_file_names[i])

        if sampling_method == MIN_OBSERVATION_COUNT_METHOD:
            _, this_metadata_dict = sample_by_min_observations(
                feature_table=this_feature_table,
                min_lead_time_sec=min_lead_time_sec,
                max_lead_time_sec=max_lead_time_sec,
                min_link_distance_metres=min_link_distance_metres,
                max_link_distance_metres=max_link_distance_metres,
                min_observation_count=min_observation_count,
                return_table=False)

        elif sampling_method == MIN_OBS_COUNT_PLUS_SAMPLING_METHOD:
            _, this_metadata_dict = sample_by_min_observations_plus(
                feature_table=this_feature_table,
                min_lead_time_sec=min_lead_time_sec,
                max_lead_time_sec=max_lead_time_sec,
                min_link_distance_metres=min_link_distance_metres,
                max_link_distance_metres=max_link_distance_metres,
                wind_speed_percentile_level=wind_speed_percentile_level,
                wind_speed_class_cutoffs_kt=wind_speed_class_cutoffs_kt,
                min_observation_count=min_observation_count,
                return_table=False)

        elif sampling_method == MIN_OBSERVATION_DENSITY_METHOD:
            _, this_metadata_dict = sample_by_min_obs_density(
                feature_table=this_feature_table,
                min_lead_time_sec=min_lead_time_sec,
                max_lead_time_sec=max_lead_time_sec,
                min_link_distance_metres=min_link_distance_metres,
                max_link_distance_metres=max_link_distance_metres,
                min_observation_density_m02=min_observation_density_m02,
                return_table=False)

        elif sampling_method == MIN_OBS_DENSITY_PLUS_SAMPLING_METHOD:
            _, this_metadata_dict = sample_by_min_obs_density_plus(
                feature_table=this_feature_table,
                min_lead_time_sec=min_lead_time_sec,
                max_lead_time_sec=max_lead_time_sec,
                min_link_distance_metres=min_link_distance_metres,
                max_link_distance_metres=max_link_distance_metres,
                wind_speed_percentile_level=wind_speed_percentile_level,
                wind_speed_class_cutoffs_kt=wind_speed_class_cutoffs_kt,
                min_observation_density_m02=min_observation_density_m02,
                return_table=False)

        live_selected_indices_by_file[i] = this_metadata_dict[
            LIVE_SELECTED_INDICES_KEY]
        live_indices_by_file[i] = this_metadata_dict[LIVE_INDICES_KEY]
        dead_indices_by_file[i] = this_metadata_dict[DEAD_INDICES_KEY]
        num_storm_objects_by_file[i] = len(this_feature_table.index)

    print '\nSelecting feature vectors from all input files...\n'

    live_selected_indices = _indices_from_file_specific_to_overall(
        indices_by_file=live_selected_indices_by_file,
        num_objects_by_file=num_storm_objects_by_file)
    live_indices = _indices_from_file_specific_to_overall(
        indices_by_file=live_indices_by_file,
        num_objects_by_file=num_storm_objects_by_file)
    dead_indices = _indices_from_file_specific_to_overall(
        indices_by_file=dead_indices_by_file,
        num_objects_by_file=num_storm_objects_by_file)

    selected_indices = _select_dead_storms_randomly(
        live_indices=live_indices, dead_indices=dead_indices,
        live_selected_indices=live_selected_indices)
    selected_indices_by_file = _indices_from_overall_to_file_specific(
        overall_indices=selected_indices,
        num_objects_by_file=num_storm_objects_by_file)

    for i in range(num_files):
        print 'Writing sampled feature vectors to: "{0:s}"...'.format(
            output_feature_file_names[i])

        this_feature_table = feature_vectors.read_feature_table(
            input_feature_file_names[i])
        feature_vectors.write_feature_table(
            this_feature_table.iloc[selected_indices_by_file[i]],
            output_feature_file_names[i])


def sample_by_uniform_wind_speed_many_files(
        input_feature_file_names, output_feature_file_names, min_lead_time_sec,
        max_lead_time_sec, min_link_distance_metres, max_link_distance_metres,
        wind_speed_percentile_level,
        sampling_cutoffs_m_s01=DEFAULT_WIND_SPEED_CUTOFFS_FOR_SAMPLING_M_S01):
    """Samples storm objects from many files.

    This method uses the "uniform_wind_speed" sampling method, discussed in the
    docstring.

    N = number of input files

    :param input_feature_file_names: length-N list of paths to input files (with
        unsampled feature vectors).
    :param output_feature_file_names: length-N list of paths to output files
        (for sampled feature vectors).
    :param min_lead_time_sec: See documentation for
        `sample_by_uniform_wind_speed`.
    :param max_lead_time_sec: See doc for `sample_by_uniform_wind_speed`.
    :param min_link_distance_metres: See doc for `sample_by_uniform_wind_speed`.
    :param max_link_distance_metres: See doc for `sample_by_uniform_wind_speed`.
    :param wind_speed_percentile_level: See doc for
        `sample_by_uniform_wind_speed`.
    :param sampling_cutoffs_m_s01: See doc for `sample_by_uniform_wind_speed`.
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

    wind_speed_categories = numpy.array([], dtype=int)
    wind_observation_counts = numpy.array([], dtype=int)
    num_storm_objects_by_file = numpy.full(num_files, -1, dtype=int)

    for i in range(num_files):
        file_system_utils.mkdir_recursive_if_necessary(
            file_name=output_feature_file_names[i])

        print 'Reading unsampled feature vectors from: "{0:s}"...'.format(
            input_feature_file_names[i])
        this_feature_table = feature_vectors.read_feature_table(
            input_feature_file_names[i])

        _, this_metadata_dict = sample_by_uniform_wind_speed(
            feature_table=this_feature_table,
            min_lead_time_sec=min_lead_time_sec,
            max_lead_time_sec=max_lead_time_sec,
            min_link_distance_metres=min_link_distance_metres,
            max_link_distance_metres=max_link_distance_metres,
            wind_speed_percentile_level=wind_speed_percentile_level,
            sampling_cutoffs_m_s01=sampling_cutoffs_m_s01)

        wind_speed_categories = numpy.concatenate((
            wind_speed_categories,
            this_metadata_dict[WIND_SPEED_CATEGORIES_KEY]))
        wind_observation_counts = numpy.concatenate((
            wind_observation_counts,
            this_metadata_dict[WIND_OBSERVATION_COUNTS_KEY]))
        num_storm_objects_by_file[i] = len(this_feature_table.index)

    selected_indices = _select_storms_uniformly_by_category(
        storm_object_categories=wind_speed_categories,
        storm_object_priorities=wind_observation_counts)
    selected_indices_by_file = _indices_from_overall_to_file_specific(
        overall_indices=selected_indices,
        num_objects_by_file=num_storm_objects_by_file)

    for i in range(num_files):
        print 'Writing sampled feature vectors to: "{0:s}"...'.format(
            output_feature_file_names[i])

        this_feature_table = feature_vectors.read_feature_table(
            input_feature_file_names[i])
        feature_vectors.write_feature_table(
            this_feature_table.iloc[selected_indices_by_file[i]],
            output_feature_file_names[i])
