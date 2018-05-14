"""Methods for handling feature vectors."""

import pickle
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import radar_statistics as radar_stats
from gewittergefahr.gg_utils import shape_statistics as shape_stats
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.gg_utils import link_events_to_storms as events2storms

STORM_COLUMNS_TO_KEEP = [
    tracking_utils.STORM_ID_COLUMN, tracking_utils.TIME_COLUMN,
    tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN,
    events2storms.STORM_END_TIME_COLUMN]

COLUMNS_TO_MERGE_ON = [
    tracking_utils.STORM_ID_COLUMN, tracking_utils.TIME_COLUMN]


def _check_labels_in_table(storm_object_table):
    """Ensures that table contains labels (target variables).

    :param storm_object_table: pandas DataFrame, where each row is one storm
        object.
    :return: label_column_names: 1-D list with names of columns containing
        labels (target variables).
    :return: num_wind_obs_column_names: 1-D list with names of columns
        containing number of wind observations used to create label.
    :raises: ValueError: if `feature_table` contains no label columns.
    """

    try:
        wind_label_column_names, num_wind_obs_column_names = (
            labels.check_wind_speed_label_table(storm_object_table))
    except TypeError:
        wind_label_column_names = []
        num_wind_obs_column_names = None

    try:
        tornado_label_column_names = labels.check_tornado_label_table(
            storm_object_table)
    except TypeError:
        tornado_label_column_names = []

    label_column_names = wind_label_column_names + tornado_label_column_names
    if not label_column_names:
        raise ValueError(
            'feature_table contains no columns with labels (target variables).')

    return label_column_names, num_wind_obs_column_names


def _get_columns_to_write(feature_table):
    """Determines which columns of feature table will be written to file.

    :param feature_table: pandas DataFrame created by
        `join_features_and_labels`.
    :return: columns_to_write: 1-D list of columns to write.
    """

    feature_column_names, label_column_names, num_wind_obs_column_names = (
        check_feature_table(feature_table))
    if num_wind_obs_column_names is None:
        num_wind_obs_column_names = []

    distance_buffer_column_names = tracking_utils.get_distance_buffer_columns(
        feature_table)
    if distance_buffer_column_names is None:
        distance_buffer_column_names = []

    return (
        feature_column_names + label_column_names + num_wind_obs_column_names +
        distance_buffer_column_names + STORM_COLUMNS_TO_KEEP)


def check_feature_table(feature_table):
    """Ensures that table contains both predictors and labels.

    Specifically, `feature_table` must contain the following columns:

    - one or more features (predictor variables)
    - one or more labels (target variables)

    :param feature_table: pandas DataFrame, where each row is one storm object.
    :return: feature_column_names: 1-D list with names of columns containing
        features (predictor variables).
    :return: label_column_names: 1-D list with names of columns containing
        labels (target variables).
    :return: num_wind_obs_column_names: 1-D list with names of columns
        containing number of wind observations used to create label.
    :raises: ValueError: if `feature_table` contains no feature columns.
    """

    error_checking.assert_columns_in_dataframe(
        feature_table, STORM_COLUMNS_TO_KEEP)

    # Find columns with features (predictor variables).
    feature_column_names = radar_stats.get_statistic_columns(feature_table)
    shape_stat_column_names = shape_stats.get_statistic_columns(feature_table)
    if shape_stat_column_names:
        if feature_column_names:
            feature_column_names += shape_stat_column_names
        else:
            feature_column_names = shape_stat_column_names

    sounding_stat_column_names = soundings.get_statistic_columns(feature_table)
    if sounding_stat_column_names:
        if feature_column_names:
            feature_column_names += sounding_stat_column_names
        else:
            feature_column_names = sounding_stat_column_names

    if not feature_column_names:
        raise ValueError('feature_table contains no columns with features '
                         '(predictor variables).')

    # Find columns with labels (target variables).
    label_column_names, num_wind_obs_column_names = (
        _check_labels_in_table(feature_table))

    return feature_column_names, label_column_names, num_wind_obs_column_names


def join_features_and_labels(
        storm_to_events_table, radar_statistic_table=None,
        shape_statistic_table=None, sounding_statistic_table=None):
    """Joins tables with features (predictor vrbles) and labels (target vrbles).

    For all tables, each row is one storm object.

    :param storm_to_events_table: pandas DataFrame created by
        `labels.label_wind_speed_for_regression`,
        `labels.label_wind_speed_for_classification`, or
        `labels.label_tornado_occurrence`.
    :param radar_statistic_table: pandas DataFrame created by
        `radar_statistics.get_storm_based_radar_stats_myrorss_or_mrms` or
        `radar_statistics.get_storm_based_radar_stats_gridrad`.
    :param shape_statistic_table: pandas DataFrame created by
        `shape_statistics.get_stats_for_storm_objects`.
    :param sounding_statistic_table: pandas DataFrame created by
        `soundings.get_sounding_stats_for_storm_objects`.
    :return: feature_table: pandas DataFrame containing columns with features
        (predictor variables), labels (target variables), and number of wind
        observations used to create label (where applicable).  Also, if
        `storm_to_events_table` contained any columns with storm-centered
        distance buffers (created by
        `storm_tracking_utils.make_buffers_around_storm_objects`), these are in
        `feature_table`.  Finally, `feature_table` always contains the following
        columns.
    feature_table.storm_id: String ID for storm cell.
    feature_table.unix_time_sec: Valid time of storm object.
    feature_table.end_time_unix_sec: End time of storm cell.
    feature_table.polygon_object_latlng: Instance of `shapely.geometry.Polygon`,
        defining the storm boundary (outline of the storm object).
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

    if sounding_statistic_table is not None:
        sounding_stat_column_names = soundings.check_sounding_statistic_table(
            sounding_statistic_table, require_storm_objects=True)
        sounding_statistic_table = sounding_statistic_table[
            COLUMNS_TO_MERGE_ON + sounding_stat_column_names]

        if feature_table is None:
            feature_table = sounding_statistic_table
        else:
            feature_table = feature_table.merge(
                sounding_statistic_table, on=COLUMNS_TO_MERGE_ON, how='inner')

    distance_buffer_column_names = tracking_utils.get_distance_buffer_columns(
        storm_to_events_table)
    if distance_buffer_column_names is None:
        distance_buffer_column_names = []

    label_column_names, num_wind_obs_column_names = _check_labels_in_table(
        storm_to_events_table)
    if num_wind_obs_column_names is None:
        num_wind_obs_column_names = []

    storm_to_events_table = storm_to_events_table[
        STORM_COLUMNS_TO_KEEP + label_column_names + num_wind_obs_column_names +
        distance_buffer_column_names]
    return feature_table.merge(
        storm_to_events_table, on=COLUMNS_TO_MERGE_ON, how='inner')


def write_feature_table(feature_table, pickle_file_name):
    """Writes feature table to Pickle file.

    :param feature_table: pandas DataFrame created by
        `join_features_and_labels`.
    :param pickle_file_name: Path to output file.
    """

    columns_to_write = _get_columns_to_write(feature_table)
    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(feature_table[columns_to_write], pickle_file_handle)
    pickle_file_handle.close()


def read_feature_table(pickle_file_name):
    """Reads feature table from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: feature_table: pandas DataFrame in format created by
        `join_features_and_labels`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    feature_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    check_feature_table(feature_table)
    return feature_table
