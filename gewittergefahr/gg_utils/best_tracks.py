"""Runs Python version of the w2besttrack algorithm in WDSS-II*.

* WDSS-II = Warning Decision Support System -- Integrated Information

For WDSS-II documentation, see:
http://www.cimms.ou.edu/~lakshman/wdssii/index.shtml

For the w2besttrack paper, see Lakshmanan et al. (2015).

--- REFERENCES ---

Lakshmanan, V., B. Herzog, and D. Kingfield, 2015: "A method for extracting
    post-event storm tracks". Journal of Applied Meteorology and Climatology,
    54 (2), 451-462.
"""

import copy
import numpy
import pandas
from sklearn.linear_model import TheilSenRegressor
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import geodetic_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

EMPTY_STORM_ID = 'no_storm'

DEFAULT_MAX_EXTRAP_TIME_SEC = 610
DEFAULT_MAX_PREDICTION_ERROR_METRES = 10000.
DEFAULT_MAX_JOIN_TIME_SEC = 915
DEFAULT_MAX_JOIN_DISTANCE_M_S01 = 30.
DEFAULT_MAX_MEAN_JOIN_ERROR_M_S01 = 10.
USE_EXTRA_BREAKUP_CRITERIA_DEFAULT_FLAG = True

DEFAULT_NUM_MAIN_ITERS = 3
DEFAULT_NUM_BREAKUP_ITERS = 3
DEFAULT_MIN_OBJECTS_IN_TRACK = 1
MIN_OBJECTS_IN_TRACK_FOR_THEA = 1

CENTROID_X_COLUMN = 'centroid_x_metres'
CENTROID_Y_COLUMN = 'centroid_y_metres'

TRACK_TIMES_COLUMN = 'unix_times_sec'
TRACK_START_TIME_COLUMN = 'start_time_unix_sec'
TRACK_END_TIME_COLUMN = 'end_time_unix_sec'
TRACK_X_COORDS_COLUMN = 'x_coords_metres'
TRACK_Y_COORDS_COLUMN = 'y_coords_metres'
OBJECT_INDICES_COLUMN_FOR_TRACK = 'object_indices'

THEIL_SEN_X_INTERCEPT_COLUMN = 'theil_sen_x_intercept_metres'
THEIL_SEN_X_VELOCITY_COLUMN = 'theil_sen_x_velocity_m_s01'
THEIL_SEN_Y_INTERCEPT_COLUMN = 'theil_sen_y_intercept_metres'
THEIL_SEN_Y_VELOCITY_COLUMN = 'theil_sen_y_velocity_m_s01'

THEIL_SEN_COLUMNS = [
    THEIL_SEN_X_INTERCEPT_COLUMN, THEIL_SEN_X_VELOCITY_COLUMN,
    THEIL_SEN_Y_INTERCEPT_COLUMN, THEIL_SEN_Y_VELOCITY_COLUMN
]

FILE_INDEX_COLUMN = 'file_index'

INPUT_COLUMNS_TO_KEEP = [
    tracking_utils.STORM_ID_COLUMN, tracking_utils.TIME_COLUMN,
    tracking_utils.CENTROID_LAT_COLUMN, tracking_utils.CENTROID_LNG_COLUMN,
    tracking_utils.GRID_POINT_LAT_COLUMN, tracking_utils.GRID_POINT_LNG_COLUMN,
    tracking_utils.GRID_POINT_ROW_COLUMN,
    tracking_utils.GRID_POINT_COLUMN_COLUMN
]

COLUMNS_TO_MERGE_ON = [
    tracking_utils.ORIG_STORM_ID_COLUMN, tracking_utils.TIME_COLUMN
]

OUTPUT_COLUMNS_FROM_BEST_TRACK = [
    tracking_utils.STORM_ID_COLUMN, tracking_utils.ORIG_STORM_ID_COLUMN,
    tracking_utils.TIME_COLUMN, tracking_utils.AGE_COLUMN,
    tracking_utils.TRACKING_START_TIME_COLUMN,
    tracking_utils.TRACKING_END_TIME_COLUMN,
    tracking_utils.CELL_START_TIME_COLUMN, tracking_utils.CELL_END_TIME_COLUMN,
    tracking_utils.GRID_POINT_LAT_COLUMN, tracking_utils.GRID_POINT_LNG_COLUMN,
    tracking_utils.GRID_POINT_ROW_COLUMN,
    tracking_utils.GRID_POINT_COLUMN_COLUMN
]

EMPTY_TRACK_AGE_SEC = -1

ATTRIBUTES_TO_OVERWRITE = [
    tracking_utils.AGE_COLUMN, tracking_utils.TRACKING_START_TIME_COLUMN,
    tracking_utils.TRACKING_END_TIME_COLUMN,
    tracking_utils.GRID_POINT_LAT_COLUMN, tracking_utils.GRID_POINT_LNG_COLUMN,
    tracking_utils.GRID_POINT_ROW_COLUMN,
    tracking_utils.GRID_POINT_COLUMN_COLUMN
]

VERTEX_LATITUDES_COLUMN = 'polygon_vertex_latitudes_deg'
VERTEX_LONGITUDES_COLUMN = 'polygon_vertex_longitudes_deg'

OUTPUT_COLUMNS_FOR_THEA = [
    tracking_utils.STORM_ID_COLUMN, tracking_utils.TIME_COLUMN,
    tracking_utils.CENTROID_LAT_COLUMN, tracking_utils.CENTROID_LNG_COLUMN,
    VERTEX_LATITUDES_COLUMN, VERTEX_LONGITUDES_COLUMN
]


def _theil_sen_fit_one_track(
        object_x_coords_metres, object_y_coords_metres, object_times_unix_sec):
    """Fits Theil-Sen model for one storm track.

    P = number of storm objects in the track

    :param object_x_coords_metres: length-P numpy array of x-coordinates.
    :param object_y_coords_metres: length-P numpy array of y-coordinates.
    :param object_times_unix_sec: length-P numpy array of times.
    :return: x_intercept_metres: x-intercept.
    :return: x_velocity_m_s01: x-velocity (metres per second).
    :return: y_intercept_metres: y-intercept.
    :return: y_velocity_m_s01: y-velocity (metres per second).
    """

    num_storm_objects = len(object_times_unix_sec)
    object_times_unix_sec = numpy.reshape(
        object_times_unix_sec, (num_storm_objects, 1))

    model_object_for_x = TheilSenRegressor(fit_intercept=True)
    model_object_for_x.fit(object_times_unix_sec, object_x_coords_metres)
    model_object_for_y = TheilSenRegressor(fit_intercept=True)
    model_object_for_y.fit(object_times_unix_sec, object_y_coords_metres)

    return (model_object_for_x.intercept_, model_object_for_x.coef_,
            model_object_for_y.intercept_, model_object_for_y.coef_)


def _theil_sen_predict(
        x_intercept_metres, x_velocity_m_s01, y_intercept_metres,
        y_velocity_m_s01, query_time_unix_sec):
    """Uses one Theil-Sen model to predict storm location at one time.

    :param x_intercept_metres: See doc for `_theil_sen_fit_one_track`.
    :param x_velocity_m_s01: Same.
    :param y_intercept_metres: Same.
    :param y_velocity_m_s01: Same.
    :param query_time_unix_sec: Query time.
    :return: x_predicted_metres: x-coordinate of predicted location.
    :return: y_predicted_metres: y-coordinate of predicted location.
    """

    x_predicted_metres = (
        x_intercept_metres + x_velocity_m_s01 * query_time_unix_sec)
    y_predicted_metres = (
        y_intercept_metres + y_velocity_m_s01 * query_time_unix_sec)
    return x_predicted_metres, y_predicted_metres


def _theil_sen_predict_many_times(
        x_intercept_metres, x_velocity_m_s01, y_intercept_metres,
        y_velocity_m_s01, query_times_unix_sec):
    """Uses one Theil-Sen model to predict storm location at many times.

    P = number of query times

    :param x_intercept_metres: See doc for `_theil_sen_fit_one_track`.
    :param x_velocity_m_s01: Same.
    :param y_intercept_metres: Same.
    :param y_velocity_m_s01: Same.
    :param query_times_unix_sec: length-P numpy array of query times.
    :return: predicted_x_coords_metres: length-P numpy array of predicted
        x-coordinates.
    :return: predicted_y_coords_metres: length-P numpy array of predicted
        y-coordinates.
    """

    predicted_x_coords_metres = (
        x_intercept_metres + x_velocity_m_s01 * query_times_unix_sec)
    predicted_y_coords_metres = (
        y_intercept_metres + y_velocity_m_s01 * query_times_unix_sec)
    return predicted_x_coords_metres, predicted_y_coords_metres


def _theil_sen_predict_many_models(
        x_intercepts_metres, x_velocities_m_s01, y_intercepts_metres,
        y_velocities_m_s01, query_time_unix_sec):
    """Uses many Theil-Sen models to predict storm location at one time.

    M = number of models

    :param x_intercepts_metres: length-M numpy array of x-intercepts (metres),
        produced by `_theil_sen_fit_one_track`.
    :param x_velocities_m_s01: length-M numpy array of x-velocities (metres per
        second), produced by `_theil_sen_fit_one_track`.
    :param y_intercepts_metres: Same as `x_intercepts_metres`, but for
        y-coordinates.
    :param y_velocities_m_s01: Same as `x_velocities_m_s01`, but for
        y-coordinates.
    :param query_time_unix_sec: Query time.
    :return: predicted_x_coords_metres: length-M numpy array of predicted
        x-coordinates.
    :return: predicted_y_coords_metres: length-M numpy array of predicted
        y-coordinates.
    """

    predicted_x_coords_metres = (
        x_intercepts_metres + x_velocities_m_s01 * query_time_unix_sec)
    predicted_y_coords_metres = (
        y_intercepts_metres + y_velocities_m_s01 * query_time_unix_sec)
    return predicted_x_coords_metres, predicted_y_coords_metres


def _get_theil_sen_errors_one_object(
        object_x_metres, object_y_metres, object_time_unix_sec,
        storm_track_table):
    """Computes Theil-Sen errors produced by many models for one storm object.

    T = number of storm tracks

    :param object_x_metres: x-coordinate of the one storm object.
    :param object_y_metres: y-coordinate of the one storm object.
    :param object_time_unix_sec: Valid time of the one storm object.
    :param storm_track_table: T-row pandas DataFrame created by
        `theil_sen_fit_for_each_track`.
    :return: error_by_track_metres: length-T numpy array with error yielded by
        the Theil-Sen model for each track.
    """

    (predicted_x_coords_metres, predicted_y_coords_metres
    ) = _theil_sen_predict_many_models(
        x_intercepts_metres=storm_track_table[
            THEIL_SEN_X_INTERCEPT_COLUMN].values,
        x_velocities_m_s01=storm_track_table[
            THEIL_SEN_X_VELOCITY_COLUMN].values,
        y_intercepts_metres=storm_track_table[
            THEIL_SEN_Y_INTERCEPT_COLUMN].values,
        y_velocities_m_s01=storm_track_table[
            THEIL_SEN_Y_VELOCITY_COLUMN].values,
        query_time_unix_sec=object_time_unix_sec)

    return numpy.sqrt(
        (object_x_metres - predicted_x_coords_metres) ** 2 +
        (object_y_metres - predicted_y_coords_metres) ** 2)


def _get_join_time_for_two_tracks(start_times_unix_sec, end_times_unix_sec):
    """Computes join time for two storm tracks.

    "Join time" = time elapsed between the end of the early track and the start
    of the late track.
    "Early track" = the one that starts first.
    "Late track" = the one that starts second.

    If the late track starts before the early track ends, this method returns
    NaN (they should not be joined).

    :param start_times_unix_sec: length-2 numpy array of start times.
    :param end_times_unix_sec: length-2 numpy array of end times.
    :return: join_time_sec: Join time.
    :return: early_index: Index of early storm (either 0 or 1).
    :return: late_index: Index of late storm (either 0 or 1).
    """

    if start_times_unix_sec[0] >= start_times_unix_sec[1]:
        early_index = 1
        late_index = 0
    else:
        early_index = 0
        late_index = 1

    join_time_sec = (
        start_times_unix_sec[late_index] - end_times_unix_sec[early_index])
    if join_time_sec <= 0:
        join_time_sec = numpy.nan

    return join_time_sec, early_index, late_index


def _get_join_distance_for_two_tracks(
        x_coords_early_metres=None, y_coords_early_metres=None,
        x_coords_late_metres=None, y_coords_late_metres=None):
    """Computes join distance for two storm tracks.

    "Join distance" = distance between the start point of the late track and end
    point of the early track.

    For the definitions of "early track" and "late track," see documentation for
    _get_join_time_for_two_tracks.

    P_e = number of points in early track
    P_l = number of points in late track

    :param x_coords_early_metres: 1-D numpy array (length P_e) with time-sorted
        x-coordinates of early track.
    :param y_coords_early_metres: 1-D numpy array (length P_e) with time-sorted
        y-coordinates of early track.
    :param x_coords_late_metres: 1-D numpy array (length P_l) with time-sorted
        x-coordinates of late track.
    :param y_coords_late_metres: 1-D numpy array (length P_l) with time-sorted
        y-coordinates of late track.
    :return: join_distance_metres: Join distance.
    """

    return numpy.sqrt(
        (x_coords_late_metres[0] - x_coords_early_metres[-1]) ** 2 +
        (y_coords_late_metres[0] - y_coords_early_metres[-1]) ** 2)


def _get_velocity_difference_two_tracks(
        storm_track_table, first_index, second_index):
    """Computes velocity difference between two storm tracks.

    :param storm_track_table: pandas DataFrame created by
        `theil_sen_fit_for_each_track`.
    :param first_index: Array index of the first track in the comparison.
    :param second_index: Array index of the second track in the comparison.
    :return: velocity_difference_m_s01: Magnitude of vector difference between
        the two velocities.
    """

    x_difference_m_s01 = (
        storm_track_table[THEIL_SEN_X_VELOCITY_COLUMN].values[first_index] -
        storm_track_table[THEIL_SEN_X_VELOCITY_COLUMN].values[second_index])
    y_difference_m_s01 = (
        storm_track_table[THEIL_SEN_Y_VELOCITY_COLUMN].values[first_index] -
        storm_track_table[THEIL_SEN_Y_VELOCITY_COLUMN].values[second_index])
    return numpy.sqrt(x_difference_m_s01 ** 2 + y_difference_m_s01 ** 2)


def _get_theil_sen_error_two_tracks(
        storm_track_table, early_track_index, late_track_index):
    """Computes Theil-Sen error from using early track to predict late track.

    :param storm_track_table: pandas DataFrame created by
        `theil_sen_fit_for_each_track`.
    :param early_track_index: Array index of the earlier track in the
        comparison.
    :param late_track_index: Array index of the later track in the comparison.
    :return: mean_error_m_s01: Mean error incurred by using the early track to
        predict locations in the late track (metres per second).
    """

    (predicted_x_coords_metres, predicted_y_coords_metres
    ) = _theil_sen_predict_many_times(
        x_intercept_metres=storm_track_table[
            THEIL_SEN_X_INTERCEPT_COLUMN].values[early_track_index],
        x_velocity_m_s01=storm_track_table[
            THEIL_SEN_X_VELOCITY_COLUMN].values[early_track_index],
        y_intercept_metres=storm_track_table[
            THEIL_SEN_Y_INTERCEPT_COLUMN].values[early_track_index],
        y_velocity_m_s01=storm_track_table[
            THEIL_SEN_Y_VELOCITY_COLUMN].values[early_track_index],
        query_times_unix_sec=storm_track_table[
            TRACK_TIMES_COLUMN].values[late_track_index])

    x_errors_metres = predicted_x_coords_metres - storm_track_table[
        TRACK_X_COORDS_COLUMN].values[late_track_index]
    y_errors_metres = predicted_y_coords_metres - storm_track_table[
        TRACK_Y_COORDS_COLUMN].values[late_track_index]

    prediction_errors_metres = numpy.sqrt(
        x_errors_metres ** 2 + y_errors_metres ** 2)
    extrap_times_seconds = (
        storm_track_table[TRACK_TIMES_COLUMN].values[late_track_index] -
        numpy.max(
            storm_track_table[TRACK_TIMES_COLUMN].values[early_track_index]
        ))
    return numpy.mean(prediction_errors_metres / extrap_times_seconds)


def _break_ties_one_track(storm_track_table, storm_track_index):
    """Breaks ties for one storm track.

    Specifically, this method ensures that there is no time step with multiple
    objects associated with the track.

    :param storm_track_table: pandas DataFrame created by
        `theil_sen_fit_for_each_track`.
    :param storm_track_index: Array index.  This method will break ties for the
        [k]th track, where k = `storm_track_index`.
    :return: object_indices_to_remove: 1-D numpy array with indices of storm
        objects to remove.  If `k in object_indices_to_remove`, the [k]th storm
        object in the track should be removed.
    """

    object_times_unix_sec = storm_track_table[
        TRACK_TIMES_COLUMN].values[storm_track_index]
    unique_times_unix_sec, orig_to_unique_indices = numpy.unique(
        object_times_unix_sec, return_inverse=True)

    if len(unique_times_unix_sec) == len(object_times_unix_sec):
        return numpy.array([], dtype=int)

    object_x_coords_metres = storm_track_table[
        TRACK_X_COORDS_COLUMN].values[storm_track_index]
    object_y_coords_metres = storm_track_table[
        TRACK_Y_COORDS_COLUMN].values[storm_track_index]

    theil_sen_x_intercept_metres = storm_track_table[
        THEIL_SEN_X_INTERCEPT_COLUMN].values[storm_track_index]
    theil_sen_x_velocity_m_s01 = storm_track_table[
        THEIL_SEN_X_VELOCITY_COLUMN].values[storm_track_index]
    theil_sen_y_intercept_metres = storm_track_table[
        THEIL_SEN_Y_INTERCEPT_COLUMN].values[storm_track_index]
    theil_sen_y_velocity_m_s01 = storm_track_table[
        THEIL_SEN_Y_VELOCITY_COLUMN].values[storm_track_index]

    num_storm_objects = len(object_times_unix_sec)
    object_indices_to_keep = numpy.linspace(
        0, num_storm_objects - 1, num=num_storm_objects, dtype=int)

    while len(unique_times_unix_sec) < len(object_times_unix_sec):
        for i in range(len(unique_times_unix_sec)):
            these_object_indices = numpy.where(orig_to_unique_indices == i)[0]
            if len(these_object_indices) == 1:  # No tie to break
                continue

            (this_x_predicted_metres, this_y_predicted_metres
            ) = _theil_sen_predict(
                x_intercept_metres=theil_sen_x_intercept_metres,
                x_velocity_m_s01=theil_sen_x_velocity_m_s01,
                y_intercept_metres=theil_sen_y_intercept_metres,
                y_velocity_m_s01=theil_sen_y_velocity_m_s01,
                query_time_unix_sec=unique_times_unix_sec[i])

            these_errors_metres = numpy.sqrt(
                (object_x_coords_metres[these_object_indices] -
                 this_x_predicted_metres) ** 2 +
                (object_y_coords_metres[these_object_indices] -
                 this_y_predicted_metres) ** 2)
            this_nearest_object_index = these_object_indices[
                numpy.argmin(these_errors_metres)]

            these_object_indices_to_remove = these_object_indices.tolist()
            these_object_indices_to_remove.remove(this_nearest_object_index)
            these_object_indices_to_remove = numpy.array(
                these_object_indices_to_remove, dtype=int)

            object_x_coords_metres = numpy.delete(
                object_x_coords_metres, these_object_indices_to_remove)
            object_y_coords_metres = numpy.delete(
                object_y_coords_metres, these_object_indices_to_remove)
            object_times_unix_sec = numpy.delete(
                object_times_unix_sec, these_object_indices_to_remove)
            object_indices_to_keep = numpy.delete(
                object_indices_to_keep, these_object_indices_to_remove)

            (theil_sen_x_intercept_metres, theil_sen_x_velocity_m_s01,
             theil_sen_y_intercept_metres, theil_sen_y_velocity_m_s01
            ) = _theil_sen_fit_one_track(
                object_x_coords_metres=object_x_coords_metres,
                object_y_coords_metres=object_y_coords_metres,
                object_times_unix_sec=object_times_unix_sec)

            break

        unique_times_unix_sec, orig_to_unique_indices = numpy.unique(
            object_times_unix_sec, return_inverse=True)

    remove_object_flags = numpy.full(num_storm_objects, True, dtype=bool)
    remove_object_flags[object_indices_to_keep] = False
    return numpy.where(remove_object_flags)[0]


def _find_changed_tracks(storm_track_table, orig_storm_track_table):
    """Finds tracks that chgd from orig_storm_track_table to storm_track_table.

    :param storm_track_table: pandas DataFrame with columns documented in
        storm_objects_to_tracks.
    :param orig_storm_track_table: pandas DataFrame with columns documented in
        storm_objects_to_tracks.
    :return: track_changed_indices: 1-D numpy array of rows (indices into
        storm_track_table) for which track changed.
    """

    num_storm_tracks = len(storm_track_table.index)
    orig_storm_ids = numpy.asarray(
        orig_storm_track_table[tracking_utils.STORM_ID_COLUMN].values)
    track_changed_flags = numpy.full(num_storm_tracks, False, dtype=bool)

    for j in range(num_storm_tracks):
        this_storm_in_orig_flags = (
            orig_storm_ids == storm_track_table[
                tracking_utils.STORM_ID_COLUMN].values[j])
        this_orig_index = numpy.where(this_storm_in_orig_flags)[0][0]

        if not numpy.array_equal(
                storm_track_table[TRACK_TIMES_COLUMN].values[j],
                orig_storm_track_table[TRACK_TIMES_COLUMN].values[
                    this_orig_index]):
            track_changed_flags[j] = True
            continue

        if not numpy.array_equal(
                storm_track_table[TRACK_X_COORDS_COLUMN].values[j],
                orig_storm_track_table[TRACK_X_COORDS_COLUMN].values[
                    this_orig_index]):
            track_changed_flags[j] = True
            continue

        if not numpy.array_equal(
                storm_track_table[TRACK_Y_COORDS_COLUMN].values[j],
                orig_storm_track_table[TRACK_Y_COORDS_COLUMN].values[
                    this_orig_index]):
            track_changed_flags[j] = True
            continue

    return numpy.where(track_changed_flags)[0]


def check_best_track_params(
        max_extrap_time_for_breakup_sec=DEFAULT_MAX_EXTRAP_TIME_SEC,
        max_prediction_error_for_breakup_metres=
        DEFAULT_MAX_PREDICTION_ERROR_METRES,
        use_extra_breakup_criteria=USE_EXTRA_BREAKUP_CRITERIA_DEFAULT_FLAG,
        max_join_time_sec=DEFAULT_MAX_JOIN_TIME_SEC,
        max_join_distance_m_s01=DEFAULT_MAX_JOIN_DISTANCE_M_S01,
        max_mean_join_error_m_s01=DEFAULT_MAX_MEAN_JOIN_ERROR_M_S01,
        num_main_iters=DEFAULT_NUM_MAIN_ITERS,
        num_breakup_iters=DEFAULT_NUM_BREAKUP_ITERS,
        min_objects_in_track=DEFAULT_MIN_OBJECTS_IN_TRACK):
    """Error-checking of best-track parameters.

    :param max_extrap_time_for_breakup_sec: See documentation for
        run_best_track.
    :param max_prediction_error_for_breakup_metres: See doc for run_best_track.
    :param use_extra_breakup_criteria: See doc for run_best_track.
    :param max_join_time_sec: See doc for run_best_track.
    :param max_join_distance_m_s01: See doc for run_best_track.
    :param max_mean_join_error_m_s01: See doc for run_best_track.
    :param num_main_iters: See doc for run_best_track.
    :param num_breakup_iters: See doc for run_best_track.
    :param min_objects_in_track: See doc for run_best_track.
    """

    error_checking.assert_is_integer(max_extrap_time_for_breakup_sec)
    error_checking.assert_is_greater(max_extrap_time_for_breakup_sec, 0)
    error_checking.assert_is_greater(
        max_prediction_error_for_breakup_metres, 0.)
    error_checking.assert_is_boolean(use_extra_breakup_criteria)

    error_checking.assert_is_integer(max_join_time_sec)
    error_checking.assert_is_greater(max_join_time_sec, 0)
    error_checking.assert_is_greater(max_join_distance_m_s01, 0.)
    error_checking.assert_is_greater(max_mean_join_error_m_s01, 0.)

    error_checking.assert_is_integer(num_main_iters)
    error_checking.assert_is_greater(num_main_iters, 0)
    error_checking.assert_is_integer(num_breakup_iters)
    error_checking.assert_is_greater(num_breakup_iters, 0)
    error_checking.assert_is_integer(min_objects_in_track)
    error_checking.assert_is_geq(min_objects_in_track, 1)


def project_centroids_latlng_to_xy(storm_object_table):
    """Projects storm centroids from lat-long to x-y coordinates.

    :param storm_object_table: pandas DataFrame with at least the following
        columns.
    storm_object_table.centroid_lat_deg: Latitude (deg N) of storm centroid.
    storm_object_table.centroid_lng_deg: Longitude (deg E) of storm centroid.

    :return: storm_object_table: Same as input, except that "centroid_lat_deg"
        and "centroid_lng_deg" are replaced by the following.
    storm_object_table.centroid_x_metres: x-coordinate of storm centroid.
    storm_object_table.centroid_y_metres: y-coordinate of storm centroid.
    """

    global_centroid_lat_deg, global_centroid_lng_deg = (
        geodetic_utils.get_latlng_centroid(
            latitudes_deg=storm_object_table[
                tracking_utils.CENTROID_LAT_COLUMN].values,
            longitudes_deg=storm_object_table[
                tracking_utils.CENTROID_LNG_COLUMN].values)
    )

    projection_object = projections.init_azimuthal_equidistant_projection(
        global_centroid_lat_deg, global_centroid_lng_deg)

    x_centroids_metres, y_centroids_metres = projections.project_latlng_to_xy(
        storm_object_table[tracking_utils.CENTROID_LAT_COLUMN].values,
        storm_object_table[tracking_utils.CENTROID_LNG_COLUMN].values,
        projection_object=projection_object, false_easting_metres=0.,
        false_northing_metres=0.)

    argument_dict = {
        CENTROID_X_COLUMN: x_centroids_metres,
        CENTROID_Y_COLUMN: y_centroids_metres
    }

    storm_object_table = storm_object_table.assign(**argument_dict)

    return storm_object_table.drop(
        [tracking_utils.CENTROID_LAT_COLUMN,
         tracking_utils.CENTROID_LNG_COLUMN], axis=1, inplace=False)


def storm_objects_to_tracks(storm_object_table, storm_ids_to_use=None):
    """Adds tracking info to storm objects.

    T = number of times (storm objects) in a given track

    :param storm_object_table: pandas DataFrame with the following columns.
        Each row is one storm object.
    storm_object_table.storm_id: String ID for storm cell.
    storm_object_table.unix_time_sec: Valid time.
    storm_object_table.centroid_x_metres: x-coordinate of storm centroid.
    storm_object_table.centroid_y_metres: y-coordinate of storm centroid.

    :param storm_ids_to_use: 1-D list of storm IDs.  Will add tracking info only
        for these storms.  If storm_ids_to_process = None, will add tracking
        info for all storms.

    :return: storm_track_table: pandas DataFrame with the following columns.
        Each row is one storm track.
    storm_track_table.storm_id: String ID.
    storm_track_table.unix_times_sec: length-T numpy array of times.
    storm_track_table.start_time_unix_sec: First time in `unix_times_sec`.
    storm_track_table.end_time_unix_sec: Last time in `unix_times_sec`.
    storm_track_table.object_indices: length-T numpy array of indices (rows in
        storm_object_table).
    storm_track_table.x_coords_metres: length-T numpy array of x-coordinates.
    storm_track_table.y_coords_metres: length-T numpy array of y-coordinates.
    """

    storm_id_by_object = numpy.asarray(
        storm_object_table[tracking_utils.STORM_ID_COLUMN].values)
    unique_storm_ids, storm_ids_object_to_unique = numpy.unique(
        storm_id_by_object, return_inverse=True)

    if storm_ids_to_use is None:
        storm_ids_to_use = unique_storm_ids.tolist()

    error_checking.assert_is_string_list(storm_ids_to_use)
    error_checking.assert_is_numpy_array(
        numpy.asarray(storm_ids_to_use), num_dimensions=1)

    storm_ids_to_use = set(storm_ids_to_use)
    if EMPTY_STORM_ID in storm_ids_to_use:
        storm_ids_to_use.remove(EMPTY_STORM_ID)
    storm_ids_to_use = list(storm_ids_to_use)

    storm_track_dict = {tracking_utils.STORM_ID_COLUMN: storm_ids_to_use}
    storm_track_table = pandas.DataFrame.from_dict(storm_track_dict)

    num_storms_to_use = len(storm_ids_to_use)
    simple_array = numpy.full(num_storms_to_use, numpy.nan, dtype=int)
    nested_array = storm_track_table[[
        tracking_utils.STORM_ID_COLUMN,
        tracking_utils.STORM_ID_COLUMN]].values.tolist()

    argument_dict = {TRACK_TIMES_COLUMN: nested_array,
                     TRACK_X_COORDS_COLUMN: nested_array,
                     TRACK_Y_COORDS_COLUMN: nested_array,
                     OBJECT_INDICES_COLUMN_FOR_TRACK: nested_array,
                     TRACK_START_TIME_COLUMN: simple_array,
                     TRACK_END_TIME_COLUMN: simple_array}
    storm_track_table = storm_track_table.assign(**argument_dict)

    num_storms_total = len(unique_storm_ids)
    for i in range(num_storms_total):
        if unique_storm_ids[i] not in storm_ids_to_use:
            continue

        these_storm_object_indices = numpy.where(
            storm_ids_object_to_unique == i)[0]
        sort_indices = numpy.argsort(
            storm_object_table[tracking_utils.TIME_COLUMN].values[
                these_storm_object_indices])
        these_storm_object_indices = these_storm_object_indices[sort_indices]

        this_table_index = storm_ids_to_use.index(unique_storm_ids[i])

        storm_track_table[TRACK_TIMES_COLUMN].values[this_table_index] = (
            storm_object_table[tracking_utils.TIME_COLUMN].values[
                these_storm_object_indices])
        storm_track_table[TRACK_X_COORDS_COLUMN].values[this_table_index] = (
            storm_object_table[CENTROID_X_COLUMN].values[
                these_storm_object_indices])
        storm_track_table[TRACK_Y_COORDS_COLUMN].values[this_table_index] = (
            storm_object_table[CENTROID_Y_COLUMN].values[
                these_storm_object_indices])
        storm_track_table[OBJECT_INDICES_COLUMN_FOR_TRACK].values[
            this_table_index] = these_storm_object_indices

        storm_track_table[TRACK_START_TIME_COLUMN].values[this_table_index] = (
            storm_track_table[TRACK_TIMES_COLUMN].values[this_table_index][0])
        storm_track_table[TRACK_END_TIME_COLUMN].values[this_table_index] = (
            storm_track_table[TRACK_TIMES_COLUMN].values[this_table_index][-1])

    return storm_track_table


def theil_sen_fit_many_tracks(
        storm_track_table, fit_indices=None, verbose=True):
    """Fits Theil-Sen model for each storm track.

    :param storm_track_table: pandas DataFrame created by
        `storm_objects_to_tracks`.
    :param fit_indices: 1-D numpy array with indices of storm tracks to
        manipulate.  If `k in fit_indices`, a Theil-Sen model will be fit for
        the [k]th track.
    :param verbose: Boolean flag.  If True, will print progress messages to
        command window.
    :return: storm_track_table: Same as input, but with extra columns listed
        below.
    storm_track_table.theil_sen_x_intercept_metres: x-intercept for Theil-Sen
        model.
    storm_track_table.theil_sen_x_velocity_m_s01: x-velocity for Theil-Sen model
        (metres per second).
    storm_track_table.theil_sen_y_intercept_metres: y-intercept for Theil-Sen
        model.
    storm_track_table.theil_sen_y_velocity_m_s01: y-velocity for Theil-Sen model
        (metres per second).
    """

    num_tracks = len(storm_track_table.index)

    if fit_indices is None:
        fit_indices = numpy.linspace(
            0, num_tracks - 1, num=num_tracks, dtype=int)

        nan_array = numpy.full(num_tracks, numpy.nan)

        argument_dict = {
            THEIL_SEN_X_INTERCEPT_COLUMN: nan_array,
            THEIL_SEN_X_VELOCITY_COLUMN: nan_array,
            THEIL_SEN_Y_INTERCEPT_COLUMN: nan_array,
            THEIL_SEN_Y_VELOCITY_COLUMN: nan_array
        }

        storm_track_table = storm_track_table.assign(**argument_dict)

    error_checking.assert_is_integer_numpy_array(fit_indices)
    error_checking.assert_is_numpy_array(fit_indices, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(fit_indices, 0)
    error_checking.assert_is_less_than_numpy_array(fit_indices, num_tracks)

    num_fits_done = 0

    for i in fit_indices:
        if verbose and numpy.mod(num_fits_done, 100) == 0:
            print 'Have fit {0:d} of {1:d} Theil-Sen models...'.format(
                num_fits_done, len(fit_indices)
            )

        (storm_track_table[THEIL_SEN_X_INTERCEPT_COLUMN].values[i],
         storm_track_table[THEIL_SEN_X_VELOCITY_COLUMN].values[i],
         storm_track_table[THEIL_SEN_Y_INTERCEPT_COLUMN].values[i],
         storm_track_table[THEIL_SEN_Y_VELOCITY_COLUMN].values[i]
        ) = _theil_sen_fit_one_track(
            object_x_coords_metres=storm_track_table[
                TRACK_X_COORDS_COLUMN].values[i],
            object_y_coords_metres=storm_track_table[
                TRACK_Y_COORDS_COLUMN].values[i],
            object_times_unix_sec=storm_track_table[
                TRACK_TIMES_COLUMN].values[i])

        num_fits_done += 1

    if verbose:
        print 'Have fit all {0:d} Theil-Sen models!'.format(len(fit_indices))

    return storm_track_table


def get_theil_sen_error_one_track(storm_track_table, storm_track_index):
    """Computes Theil-Sen error for one storm track.

    :param storm_track_table: pandas DataFrame created by
        `theil_sen_fit_for_each_track`.
    :param storm_track_index: Array index.  This method will compute the error
        for the [k]th track, where k = `storm_track_index`.
    :return: rmse_metres: Root mean squared error of storm locations predicted
        by Theil-Sen model.  The mean is taken over all points in the track.
    """

    predicted_x_coords_metres, predicted_y_coords_metres = (
        _theil_sen_predict_many_times(
            x_intercept_metres=storm_track_table[
                THEIL_SEN_X_INTERCEPT_COLUMN].values[storm_track_index],
            x_velocity_m_s01=storm_track_table[
                THEIL_SEN_X_VELOCITY_COLUMN].values[storm_track_index],
            y_intercept_metres=storm_track_table[
                THEIL_SEN_Y_INTERCEPT_COLUMN].values[storm_track_index],
            y_velocity_m_s01=storm_track_table[
                THEIL_SEN_Y_VELOCITY_COLUMN].values[storm_track_index],
            query_times_unix_sec=storm_track_table[
                TRACK_TIMES_COLUMN].values[storm_track_index]
        )
    )

    x_errors_metres = predicted_x_coords_metres - storm_track_table[
        TRACK_X_COORDS_COLUMN].values[storm_track_index]
    y_errors_metres = predicted_y_coords_metres - storm_track_table[
        TRACK_Y_COORDS_COLUMN].values[storm_track_index]

    return numpy.sqrt(numpy.mean(x_errors_metres ** 2 + y_errors_metres ** 2))


def break_storm_tracks(
        storm_object_table=None, storm_track_table=None,
        working_object_indices=None,
        max_extrapolation_time_sec=DEFAULT_MAX_EXTRAP_TIME_SEC,
        max_prediction_error_metres=DEFAULT_MAX_PREDICTION_ERROR_METRES,
        use_extra_criteria=USE_EXTRA_BREAKUP_CRITERIA_DEFAULT_FLAG,
        min_objects_in_track=None):
    """Breaks storm tracks and reassigns storm objects.

    This is the "break-up" step in w2besttrack.

    :param storm_object_table: pandas DataFrame with the following columns.
        Each row is one storm object.
    storm_object_table.storm_id: String ID for storm cell.
    storm_object_table.unix_time_sec: Valid time.
    storm_object_table.centroid_x_metres: x-coordinate of storm centroid.
    storm_object_table.centroid_y_metres: y-coordinate of storm centroid.

    :param storm_track_table: pandas DataFrame with columns listed in
        `theil_sen_fit_many_tracks`.
    :param max_extrapolation_time_sec: Maximum extrapolation time.  No storm
        track will be extrapolated > `max_extrapolation_time_sec` before the
        time of its first storm object or after the time of its last storm
        object.
    :param working_object_indices: 1-D numpy array with indices of storm objects
        to work on (for which parent track may be changed).  If
        working_object_indices = None, this method will work on all storm
        objects.
    :param max_prediction_error_metres: Maximum prediction error.  For storm
        object s to be assigned to track S, the Theil-Sen prediction for S must
        be within `max_prediction_error_metres` of the true position of s.
    :param use_extra_criteria: Boolean flag.  If True, will use the following
        extra criterion.  If storm object s comes from a track S with
        `min_objects_in_track` storm objects, and S is the first- or second-best
        match for s, s will not be reassigned.
    :param min_objects_in_track: See above.
    :return: storm_object_table: Same as input, except that "storm_id" column
        may have changed for several objects.
    :return: storm_track_table: Same as input, except that values may have
        changed for several tracks.
    """

    error_checking.assert_is_boolean(use_extra_criteria)
    if use_extra_criteria:
        check_best_track_params(
            max_extrap_time_for_breakup_sec=max_extrapolation_time_sec,
            max_prediction_error_for_breakup_metres=max_prediction_error_metres,
            use_extra_breakup_criteria=use_extra_criteria,
            min_objects_in_track=min_objects_in_track)
    else:
        check_best_track_params(
            max_extrap_time_for_breakup_sec=max_extrapolation_time_sec,
            max_prediction_error_for_breakup_metres=max_prediction_error_metres)

    num_storm_objects = len(storm_object_table.index)
    if working_object_indices is None:
        working_object_indices = numpy.linspace(
            0, num_storm_objects - 1, num=num_storm_objects, dtype=int)

    error_checking.assert_is_integer_numpy_array(working_object_indices)
    error_checking.assert_is_numpy_array(
        working_object_indices, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(working_object_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        working_object_indices, num_storm_objects)

    num_working_objects = len(working_object_indices)
    num_objects_done = 0
    orig_storm_ids = numpy.asarray(
        storm_object_table[tracking_utils.STORM_ID_COLUMN].values[
            working_object_indices])

    for i in working_object_indices:
        if numpy.mod(num_objects_done, 1000) == 0:
            print ('Have performed break-up step for ' + str(num_objects_done) +
                   '/' + str(num_working_objects) + ' storm objects...')

        num_objects_done += 1

        times_before_start_sec = (
            storm_track_table[TRACK_START_TIME_COLUMN].values -
            storm_object_table[tracking_utils.TIME_COLUMN].values[i])
        times_after_end_sec = (
            storm_object_table[tracking_utils.TIME_COLUMN].values[i] -
            storm_track_table[TRACK_END_TIME_COLUMN].values)

        try_track_flags = numpy.logical_and(
            times_before_start_sec <= max_extrapolation_time_sec,
            times_after_end_sec <= max_extrapolation_time_sec)
        if not numpy.any(try_track_flags):
            continue

        try_track_indices = numpy.where(try_track_flags)[0]
        prediction_errors_metres = _get_theil_sen_errors_one_object(
            object_x_metres=storm_object_table[CENTROID_X_COLUMN].values[i],
            object_y_metres=storm_object_table[CENTROID_Y_COLUMN].values[i],
            object_time_unix_sec=storm_object_table[
                tracking_utils.TIME_COLUMN].values[i],
            storm_track_table=storm_track_table.iloc[try_track_indices])

        if numpy.min(prediction_errors_metres) > max_prediction_error_metres:
            continue

        if use_extra_criteria:
            orig_storm_id = storm_object_table[
                tracking_utils.STORM_ID_COLUMN].values[i]
            orig_track_indices = numpy.where(
                numpy.array(
                    storm_object_table[tracking_utils.STORM_ID_COLUMN].values)
                == orig_storm_id)[0]

            num_objects_in_orig_track = len(orig_track_indices)
            if num_objects_in_orig_track < min_objects_in_track:
                continue

            sort_indices = numpy.argsort(prediction_errors_metres)
            min_error_indices = sort_indices[0:2]
            nearest_track_indices = try_track_indices[min_error_indices]
            nearest_track_ids = storm_object_table[
                tracking_utils.STORM_ID_COLUMN].values[nearest_track_indices]

            if orig_storm_id in nearest_track_ids:
                continue

        nearest_track_index = try_track_indices[
            numpy.argmin(prediction_errors_metres)]
        storm_object_table[tracking_utils.STORM_ID_COLUMN].values[i] = (
            storm_track_table[tracking_utils.STORM_ID_COLUMN].values[
                nearest_track_index])

    print ('Have performed break-up step for all ' + str(num_working_objects) +
           ' storm objects!')

    new_storm_ids = numpy.asarray(
        storm_object_table[tracking_utils.STORM_ID_COLUMN].values[
            working_object_indices])
    object_changed_indices = numpy.where(orig_storm_ids != new_storm_ids)[0]
    print ('Assigned ' + str(len(object_changed_indices)) +
           ' storm objects to a new track during this procedure.\n\n')

    orig_storm_track_table = copy.deepcopy(storm_track_table)
    storm_track_table = storm_objects_to_tracks(storm_object_table)
    storm_track_table = storm_track_table.merge(
        orig_storm_track_table[
            THEIL_SEN_COLUMNS + [tracking_utils.STORM_ID_COLUMN]],
        on=tracking_utils.STORM_ID_COLUMN, how='left')

    track_changed_indices = _find_changed_tracks(
        storm_track_table, orig_storm_track_table)
    storm_track_table = theil_sen_fit_many_tracks(
        storm_track_table=storm_track_table, fit_indices=track_changed_indices,
        verbose=False)

    return storm_object_table, storm_track_table


def merge_storm_tracks(
        storm_object_table=None, storm_track_table=None,
        working_track_indices=None, max_join_time_sec=DEFAULT_MAX_JOIN_TIME_SEC,
        max_join_distance_m_s01=DEFAULT_MAX_JOIN_DISTANCE_M_S01,
        max_mean_prediction_error_m_s01=DEFAULT_MAX_MEAN_JOIN_ERROR_M_S01):
    """Merges pairs of similar storm tracks.

    This is the "merger" step in w2besttrack.

    :param storm_object_table: See documentation for break_storm_tracks.
    :param storm_track_table: pandas DataFrame with columns listed in
        `theil_sen_fit_many_tracks`.
    :param working_track_indices: 1-D numpy array with indices of storm tracks
        to work on (consider for merging).  If working_track_indices = None,
        this method will work on all storm tracks.
    :param max_join_time_sec: Maximum time between tracks (specifically, between
        end time of early track and start time of late track).
    :param max_join_distance_m_s01: Max distance per unit time (metres per
        second) between tracks.  Specifically, this is max distance per unit
        time between last position of early track and first position of late
        track.  Max distance = `max_join_distance_m_s01 * join_time_sec`, where
        `join_time_sec` is time elapsed between end of early track and start of
        late track.
    :param max_mean_prediction_error_m_s01: Maximum mean error per unit time
        (metres per second) generated by using early track to predict positions
        in late track.  `mean_prediction_error_m_s01` is a mean over all points
        in the late track, and `max_mean_prediction_error_m_s01` is an upper
        bound on `mean_prediction_error_m_s01`.
    :return: storm_object_table: Same as input, except that "storm_id" column
        may have changed for several objects.
    :return: storm_track_table: Same as input, except that values may have
        changed for several tracks.
    """

    check_best_track_params(
        max_join_time_sec=max_join_time_sec,
        max_join_distance_m_s01=max_join_distance_m_s01,
        max_mean_join_error_m_s01=max_mean_prediction_error_m_s01)

    num_storm_tracks = len(storm_track_table)
    if working_track_indices is None:
        working_track_indices = numpy.linspace(
            0, num_storm_tracks - 1, num=num_storm_tracks, dtype=int)

    error_checking.assert_is_integer_numpy_array(working_track_indices)
    error_checking.assert_is_numpy_array(
        working_track_indices, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(working_track_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        working_track_indices, num_storm_tracks)

    num_working_tracks = len(working_track_indices)
    num_pairs_merged = 0
    num_tracks_considered = 0
    remove_storm_track_flags = numpy.full(num_storm_tracks, False, dtype=bool)

    for j in working_track_indices:
        if numpy.mod(num_tracks_considered, 100) == 0:
            print ('Have considered ' + str(num_tracks_considered) + '/' +
                   str(num_working_tracks) + ' storm tracks for merging...')

        num_tracks_considered += 1
        this_num_objects = len(storm_track_table[TRACK_TIMES_COLUMN].values[j])
        if this_num_objects < 2:
            continue

        for k in range(j):
            if remove_storm_track_flags[k]:
                continue
            this_num_objects = len(
                storm_track_table[TRACK_TIMES_COLUMN].values[k])
            if this_num_objects < 2:
                continue

            these_track_indices = numpy.array([j, k])
            this_join_time_sec, early_index, late_index = (
                _get_join_time_for_two_tracks(
                    storm_track_table[TRACK_START_TIME_COLUMN].values[
                        these_track_indices],
                    storm_track_table[TRACK_END_TIME_COLUMN].values[
                        these_track_indices]))

            if not this_join_time_sec <= max_join_time_sec:
                continue

            early_index = these_track_indices[early_index]
            late_index = these_track_indices[late_index]

            this_join_distance_metres = _get_join_distance_for_two_tracks(
                x_coords_early_metres=
                storm_track_table[TRACK_X_COORDS_COLUMN].values[early_index],
                y_coords_early_metres=
                storm_track_table[TRACK_Y_COORDS_COLUMN].values[early_index],
                x_coords_late_metres=
                storm_track_table[TRACK_X_COORDS_COLUMN].values[late_index],
                y_coords_late_metres=
                storm_track_table[TRACK_Y_COORDS_COLUMN].values[late_index])

            this_join_distance_m_s01 = (
                this_join_distance_metres / this_join_time_sec)
            if not this_join_distance_m_s01 <= max_join_distance_m_s01:
                # print 'Join distance = {0:.1f} m/s'.format(
                #     this_join_distance_m_s01)
                continue

            this_mean_prediction_error_m_s01 = _get_theil_sen_error_two_tracks(
                storm_track_table=storm_track_table,
                early_track_index=early_index, late_track_index=late_index)

            if not (this_mean_prediction_error_m_s01 <=
                    max_mean_prediction_error_m_s01):
                # print 'Join distance = {0:.1f} m/s'.format(
                #     this_join_distance_m_s01)
                # print 'Mean prediction error = {0:.1f} m/s'.format(
                #     this_mean_prediction_error_m_s01)
                continue

            remove_storm_track_flags[k] = True
            num_pairs_merged += 1

            storm_id_j = storm_track_table[
                tracking_utils.STORM_ID_COLUMN].values[j]
            storm_id_k = storm_track_table[
                tracking_utils.STORM_ID_COLUMN].values[k]

            for i in range(len(storm_object_table.index)):
                if storm_object_table[
                        tracking_utils.STORM_ID_COLUMN].values[i] == storm_id_k:
                    storm_object_table[
                        tracking_utils.STORM_ID_COLUMN].values[i] = storm_id_j

            storm_track_table_j_only = storm_objects_to_tracks(
                storm_object_table, [storm_id_j])
            storm_track_table_j_only = theil_sen_fit_many_tracks(
                storm_track_table=storm_track_table_j_only, verbose=False)
            storm_track_table.iloc[j] = copy.deepcopy(
                storm_track_table_j_only.iloc[0])

    print ('Have considered all ' + str(num_working_tracks) +
           ' storm tracks for merging!')
    print ('Merged ' + str(num_pairs_merged) +
           ' pairs of storm tracks during this procedure.\n\n')

    remove_storm_track_rows = numpy.where(remove_storm_track_flags)[0]
    storm_track_table.drop(
        storm_track_table.index[remove_storm_track_rows], axis=0, inplace=True)

    return storm_object_table, storm_track_table


def break_ties_among_storm_objects(
        storm_object_table=None, storm_track_table=None,
        working_track_indices=None):
    """Breaks ties among storm objects.

    This is the "tie-breaker" step in w2besttrack.

    A "tie" occurs when several storm objects at the same time are assigned to
    the same track S.  This method keeps the storm object for which S has the
    smallest Theil-Sen prediction error.  All other storm objects are unassigned
    from S (and left with no track).

    :param storm_object_table: See documentation for break_storm_tracks.
    :param storm_track_table: pandas DataFrame with columns listed in
        `theil_sen_fit_many_tracks`.
    :param working_track_indices: 1-D numpy array with indices of storm tracks
        to work on (consider for tie-breaking).  If working_track_indices =
        None, this method will work on all storm tracks.
    :return: storm_object_table: Same as input, except that "storm_id" column
        may have changed for several objects.
    :return: storm_track_table: Same as input, except that values may have
        changed for several tracks.
    """

    num_storm_tracks = len(storm_track_table)
    if working_track_indices is None:
        working_track_indices = numpy.linspace(
            0, num_storm_tracks - 1, num=num_storm_tracks, dtype=int)

    error_checking.assert_is_integer_numpy_array(working_track_indices)
    error_checking.assert_is_numpy_array(
        working_track_indices, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(working_track_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        working_track_indices, num_storm_tracks)

    num_working_tracks = len(working_track_indices)
    num_ties_broken = 0
    num_tracks_with_ties_broken = 0
    num_tracks_considered = 0

    for j in working_track_indices:
        if numpy.mod(num_tracks_considered, 500) == 0:
            print ('Have considered ' + str(num_tracks_considered) + '/' +
                   str(num_working_tracks) +
                   ' storm tracks for tie-breaking...')

        num_tracks_considered += 1
        these_object_indices_to_remove = _break_ties_one_track(
            storm_track_table=storm_track_table, storm_track_index=j)
        if not len(these_object_indices_to_remove):
            continue

        these_object_indices_to_remove = storm_track_table[
            OBJECT_INDICES_COLUMN_FOR_TRACK].values[j][
                these_object_indices_to_remove]

        num_ties_broken += len(these_object_indices_to_remove)
        num_tracks_with_ties_broken += 1
        for i in these_object_indices_to_remove:
            storm_object_table[tracking_utils.STORM_ID_COLUMN].values[
                i] = EMPTY_STORM_ID

        storm_id_j = storm_track_table[
            tracking_utils.STORM_ID_COLUMN].values[j]
        storm_track_table_j_only = storm_objects_to_tracks(
            storm_object_table, [storm_id_j])
        storm_track_table_j_only = theil_sen_fit_many_tracks(
            storm_track_table=storm_track_table_j_only, verbose=False)
        storm_track_table.iloc[j] = copy.deepcopy(
            storm_track_table_j_only.iloc[0])

    print ('Have considered all ' + str(num_working_tracks) +
           ' storm tracks for tie-breaking!')
    print ('Broke ' + str(num_ties_broken) + ' ties in ' +
           str(num_tracks_with_ties_broken) +
           ' tracks during this procedure.\n\n')

    return storm_object_table, storm_track_table


def remove_short_tracks(
        storm_object_table, min_objects_in_track=DEFAULT_MIN_OBJECTS_IN_TRACK):
    """Removes storm tracks without enough storm objects.

    :param storm_object_table: pandas DataFrame with the following columns.
        Each row is one storm object.
    storm_object_table.storm_id: String ID for storm cell.
    storm_object_table.unix_time_sec: Valid time.

    :param min_objects_in_track: Minimum number of storm objects in a track.
    :return: storm_object_table: Same as input, except that all objects
        belonging to a track with < `min_objects_in_track` objects have been
        removed.
    """

    check_best_track_params(min_objects_in_track=min_objects_in_track)

    storm_id_by_object = numpy.asarray(
        storm_object_table[tracking_utils.STORM_ID_COLUMN].values)
    unique_storm_ids, storm_ids_object_to_unique = numpy.unique(
        storm_id_by_object, return_inverse=True)
    remove_storm_object_rows = numpy.array([], dtype=int)

    for i in range(len(unique_storm_ids)):
        these_object_indices = numpy.where(storm_ids_object_to_unique == i)[0]
        if (len(these_object_indices) >= min_objects_in_track and
                unique_storm_ids[i] != EMPTY_STORM_ID):
            continue

        remove_storm_object_rows = numpy.concatenate((
            remove_storm_object_rows, these_object_indices))

    return storm_object_table.drop(
        storm_object_table.index[remove_storm_object_rows], axis=0,
        inplace=False)


def get_storm_ages(
        storm_object_table, best_track_start_time_unix_sec,
        best_track_end_time_unix_sec, max_extrap_time_for_breakup_sec,
        max_join_time_sec):
    """For each storm object, computes age of corresponding storm cell.

    :param storm_object_table: pandas DataFrame with at least the following
        columns.
    storm_object_table.storm_id: String ID for storm cell.
    storm_object_table.unix_time_sec: Valid time.

    :param best_track_start_time_unix_sec: Start of tracking period.
    :param best_track_end_time_unix_sec: End of tracking period.
    :param max_extrap_time_for_breakup_sec: See documentation for
        `break_storm_tracks`.
    :param max_join_time_sec: See documentation for `merge_storm_tracks`.
    :return: storm_object_table: Same as input, but with the following extra
        columns.
    storm_object_table.age_sec: Age of storm cell.
    storm_object_table.tracking_start_time_unix_sec: Start of tracking period.
    storm_object_table.tracking_end_time_unix_sec: End of tracking period.
    storm_object_table.cell_start_time_unix_sec: Start time of storm cell (first
        time in track).
    storm_object_table.cell_end_time_unix_sec: End time of storm cell (last time
        in track).
    """

    # TODO(thunderhoser): Move this method to somewhere more general.

    error_checking.assert_is_integer(best_track_start_time_unix_sec)
    error_checking.assert_is_integer(best_track_end_time_unix_sec)
    error_checking.assert_is_greater(
        best_track_end_time_unix_sec, best_track_start_time_unix_sec)

    check_best_track_params(
        max_extrap_time_for_breakup_sec=max_extrap_time_for_breakup_sec,
        max_join_time_sec=max_join_time_sec)

    num_storm_objects = len(storm_object_table.index)
    tracking_start_times_unix_sec = numpy.full(
        num_storm_objects, best_track_start_time_unix_sec, dtype=int)
    tracking_end_times_unix_sec = numpy.full(
        num_storm_objects, best_track_end_time_unix_sec, dtype=int)
    track_ages_sec = numpy.full(
        num_storm_objects, EMPTY_TRACK_AGE_SEC, dtype=int)
    cell_start_times_unix_sec = numpy.full(num_storm_objects, -1, dtype=int)
    cell_end_times_unix_sec = numpy.full(num_storm_objects, -1, dtype=int)

    storm_id_by_object = numpy.asarray(
        storm_object_table[tracking_utils.STORM_ID_COLUMN].values)
    storm_id_by_cell, storm_ids_object_to_unique = numpy.unique(
        storm_id_by_object, return_inverse=True)

    num_storm_cells = len(storm_id_by_cell)
    max_time_offset_sec = max(
        [max_extrap_time_for_breakup_sec, max_join_time_sec])
    age_invalid_before_unix_sec = (
        best_track_start_time_unix_sec + max_time_offset_sec)
    age_invalid_after_unix_sec = (
        best_track_end_time_unix_sec - max_time_offset_sec)

    for j in range(num_storm_cells):
        these_object_indices = numpy.where(storm_ids_object_to_unique == j)[0]
        this_start_time_unix_sec = numpy.min(
            storm_object_table[tracking_utils.TIME_COLUMN].values[
                these_object_indices])
        this_end_time_unix_sec = numpy.max(
            storm_object_table[tracking_utils.TIME_COLUMN].values[
                these_object_indices])

        cell_start_times_unix_sec[
            these_object_indices] = this_start_time_unix_sec
        cell_end_times_unix_sec[these_object_indices] = this_end_time_unix_sec

        if this_start_time_unix_sec < age_invalid_before_unix_sec:
            continue
        if this_end_time_unix_sec > age_invalid_after_unix_sec:
            continue

        track_ages_sec[these_object_indices] = (
            storm_object_table[tracking_utils.TIME_COLUMN].values[
                these_object_indices] - this_start_time_unix_sec)

    argument_dict = {
        tracking_utils.TRACKING_START_TIME_COLUMN:
            tracking_start_times_unix_sec,
        tracking_utils.TRACKING_END_TIME_COLUMN: tracking_end_times_unix_sec,
        tracking_utils.AGE_COLUMN: track_ages_sec,
        tracking_utils.CELL_START_TIME_COLUMN: cell_start_times_unix_sec,
        tracking_utils.CELL_END_TIME_COLUMN: cell_end_times_unix_sec
    }
    return storm_object_table.assign(**argument_dict)


def run_best_track(
        storm_object_table,
        max_extrap_time_for_breakup_sec=DEFAULT_MAX_EXTRAP_TIME_SEC,
        max_prediction_error_for_breakup_metres=
        DEFAULT_MAX_PREDICTION_ERROR_METRES,
        use_extra_breakup_criteria=USE_EXTRA_BREAKUP_CRITERIA_DEFAULT_FLAG,
        max_join_time_sec=DEFAULT_MAX_JOIN_TIME_SEC,
        max_join_distance_m_s01=DEFAULT_MAX_JOIN_DISTANCE_M_S01,
        max_mean_join_error_m_s01=DEFAULT_MAX_MEAN_JOIN_ERROR_M_S01,
        num_main_iters=DEFAULT_NUM_MAIN_ITERS,
        num_breakup_iters=DEFAULT_NUM_BREAKUP_ITERS,
        min_objects_in_track=DEFAULT_MIN_OBJECTS_IN_TRACK):
    """Runs the full w2besttrack algorithm without IO.

    In other words, original tracks must be read before this method and new
    tracks must be written after this method.

    :param storm_object_table: See documentation for read_input_storm_objects.
    :param max_extrap_time_for_breakup_sec: See doc for break_storm_tracks.
    :param max_prediction_error_for_breakup_metres: See doc for
        break_storm_tracks.
    :param use_extra_breakup_criteria: See doc for break_storm_tracks.
    :param max_join_time_sec: See documentation for merge_storm_tracks.
    :param max_join_distance_m_s01: See documentation for merge_storm_tracks.
    :param max_mean_join_error_m_s01: See doc for merge_storm_tracks.
    :param num_main_iters: Number of main iterations (outer loops).  Each main
        iteration consists of `num_breakup_iters` break-up iterations, one
        merger iteration, and one tie-breaking iteration.
    :param num_breakup_iters: Number of break-up iterations (inner loops).
    :param min_objects_in_track: Minimum number of storm objects in a track.
        After all the main iterations, any track with < `min_objects_in_track`
        storm objects will be removed.
    :return: storm_object_table: See documentation for
        write_output_storm_objects.
    """

    check_best_track_params(
        max_extrap_time_for_breakup_sec=max_extrap_time_for_breakup_sec,
        max_prediction_error_for_breakup_metres=
        max_prediction_error_for_breakup_metres,
        use_extra_breakup_criteria=use_extra_breakup_criteria,
        max_join_time_sec=max_join_time_sec,
        max_join_distance_m_s01=max_join_distance_m_s01,
        max_mean_join_error_m_s01=max_mean_join_error_m_s01,
        num_main_iters=num_main_iters, num_breakup_iters=num_breakup_iters,
        min_objects_in_track=min_objects_in_track)

    storm_object_table = project_centroids_latlng_to_xy(
        storm_object_table)
    storm_track_table = storm_objects_to_tracks(storm_object_table)
    storm_track_table = theil_sen_fit_many_tracks(
        storm_track_table=storm_track_table, verbose=True)

    for i in range(num_main_iters):
        print ('Starting main iteration ' + str(i + 1) + '/' +
               str(num_main_iters) + '...\n\n')

        for j in range(num_breakup_iters):
            print ('Starting break-up iteration ' + str(j + 1) + '/' +
                   str(num_breakup_iters) + '...\n\n')

            storm_object_table, storm_track_table = break_storm_tracks(
                storm_object_table=storm_object_table,
                storm_track_table=storm_track_table,
                max_extrapolation_time_sec=max_extrap_time_for_breakup_sec,
                max_prediction_error_metres=
                max_prediction_error_for_breakup_metres,
                use_extra_criteria=use_extra_breakup_criteria,
                min_objects_in_track=min_objects_in_track)

        storm_object_table, storm_track_table = merge_storm_tracks(
            storm_object_table=storm_object_table,
            storm_track_table=storm_track_table,
            max_join_time_sec=max_join_time_sec,
            max_join_distance_m_s01=max_join_distance_m_s01,
            max_mean_prediction_error_m_s01=max_mean_join_error_m_s01)

        storm_object_table, storm_track_table = break_ties_among_storm_objects(
            storm_object_table, storm_track_table)

    best_track_start_time_unix_sec = numpy.min(
        storm_object_table[tracking_utils.TIME_COLUMN].values)
    best_track_end_time_unix_sec = numpy.max(
        storm_object_table[tracking_utils.TIME_COLUMN].values)

    print ('Removing storm tracks with < ' + str(min_objects_in_track) +
           ' objects...')
    storm_object_table = remove_short_tracks(
        storm_object_table, min_objects_in_track=min_objects_in_track)

    print 'Computing storm ages...'
    return get_storm_ages(
        storm_object_table=storm_object_table,
        best_track_start_time_unix_sec=best_track_start_time_unix_sec,
        best_track_end_time_unix_sec=best_track_end_time_unix_sec,
        max_extrap_time_for_breakup_sec=max_extrap_time_for_breakup_sec,
        max_join_time_sec=max_join_time_sec)


def read_input_storm_objects(input_file_names, keep_spc_date=False):
    """Reads input storm objects from one or more files.

    Input files should be in the format produced by
    `storm_tracking_io.write_processed_file`.

    P = number of grid points in a given storm object

    :param input_file_names: 1-D list of paths to input files.
    :param keep_spc_date: Boolean flag.  If True, will keep the column
        "spc_date_unix_sec" in the input files.  If False, will throw it out.
    :return: storm_object_table: pandas DataFrame with the following columns.
        Each row is one storm object.
    storm_object_table.storm_id: String ID for storm cell.
    storm_object_table.original_storm_id: Original ID (before best-track).
    storm_object_table.unix_time_sec: Valid time.
    storm_object_table.file_index: Array index of file containing storm object.
        This is an index into `input_file_names`.
    storm_object_table.centroid_lat_deg: Latitude (deg N) of storm centroid.
    storm_object_table.centroid_lng_deg: Longitude (deg E) of storm centroid.
    storm_object_table.grid_point_latitudes_deg: length-P numpy array with
        latitudes (deg N) of grid points in storm object.
    storm_object_table.grid_point_longitudes_deg: length-P numpy array with
        longitudes (deg E) of grid points in storm object.
    storm_object_table.grid_point_rows: length-P numpy array with row indices
        (integers) of grid points in storm object.
    storm_object_table.grid_point_columns: length-P numpy array with column
        indices (integers) of grid points in storm object.
    """

    error_checking.assert_is_string_list(input_file_names)
    error_checking.assert_is_numpy_array(
        numpy.asarray(input_file_names), num_dimensions=1)

    error_checking.assert_is_boolean(keep_spc_date)
    if keep_spc_date:
        columns_to_keep = INPUT_COLUMNS_TO_KEEP + [
            tracking_utils.SPC_DATE_COLUMN]
    else:
        columns_to_keep = copy.deepcopy(INPUT_COLUMNS_TO_KEEP)

    file_indices = numpy.array([])
    num_files = len(input_file_names)
    list_of_storm_object_tables = [None] * num_files

    for i in range(num_files):
        print ('Reading storm-object file ' + str(i + 1) + ' of ' +
               str(num_files) + ': ' + input_file_names[i] + '...')

        list_of_storm_object_tables[i] = tracking_io.read_processed_file(
            input_file_names[i])[columns_to_keep]

        this_num_storm_objects = len(list_of_storm_object_tables[i].index)
        file_indices = numpy.concatenate((
            file_indices,
            numpy.linspace(i, i, num=this_num_storm_objects, dtype=int)))

        if i == 0:
            continue

        list_of_storm_object_tables[i], _ = (
            list_of_storm_object_tables[i].align(
                list_of_storm_object_tables[0], axis=1))

    print '\n'
    storm_object_table = pandas.concat(
        list_of_storm_object_tables, axis=0, ignore_index=True)

    argument_dict = {
        FILE_INDEX_COLUMN: file_indices,
        tracking_utils.ORIG_STORM_ID_COLUMN:
            storm_object_table[tracking_utils.STORM_ID_COLUMN].values}
    return storm_object_table.assign(**argument_dict)


def write_output_storm_objects(
        storm_object_table, input_file_names=None, output_file_names=None):
    """Writes output storm objects (after best-track) to one or more files.

    N = number of input files = number of output files
    P = number of grid points in a given storm object

    :param storm_object_table: pandas DataFrame with the following columns.
        Each row is one storm object.
    storm_object_table.storm_id: String ID for storm cell.
    storm_object_table.original_storm_id: Original ID (before best-track).
    storm_object_table.unix_time_sec: Valid time.
    storm_object_table.file_index: Array index of input file containing storm
        object.  This is an index into `input_file_names`.
    storm_object_table.age_sec: Age of storm cell (seconds).
    storm_object_table.tracking_start_time_unix_sec: Start of tracking period.
    storm_object_table.tracking_end_time_unix_sec: End of tracking period.
    storm_object_table.cell_start_time_unix_sec: Start time of storm cell (first
        time in track).
    storm_object_table.cell_end_time_unix_sec: End time of storm cell (last time
        in track).
    storm_object_table.grid_point_latitudes_deg: length-P numpy array with
        latitudes (deg N) of grid points in storm object.
    storm_object_table.grid_point_longitudes_deg: length-P numpy array with
        longitudes (deg E) of grid points in storm object.
    storm_object_table.grid_point_rows: length-P numpy array with row indices
        (integers) of grid points in storm object.
    storm_object_table.grid_point_columns: length-P numpy array with column
        indices (integers) of grid points in storm object.

    :param input_file_names: length-N list of paths to input files (used by
        read_input_storm_objects).
    :param output_file_names: length-N list of paths to output files.
    """

    error_checking.assert_is_string_list(input_file_names)
    error_checking.assert_is_numpy_array(
        numpy.asarray(input_file_names), num_dimensions=1)
    num_files = len(input_file_names)

    error_checking.assert_is_string_list(output_file_names)
    error_checking.assert_is_numpy_array(
        numpy.asarray(output_file_names),
        exact_dimensions=numpy.array([num_files]))

    max_file_index = numpy.max(storm_object_table[FILE_INDEX_COLUMN].values)
    error_checking.assert_is_less_than(max_file_index, num_files)

    for i in range(num_files):
        print ('Writing storm-object file ' + str(i + 1) + ' of ' +
               str(num_files) + ': ' + output_file_names[i] + '...')

        file_system_utils.mkdir_recursive_if_necessary(
            file_name=output_file_names[i])

        this_input_table = tracking_io.read_processed_file(input_file_names[i])
        column_dict_old_to_new = {
            tracking_utils.STORM_ID_COLUMN: tracking_utils.ORIG_STORM_ID_COLUMN}
        this_input_table.rename(columns=column_dict_old_to_new, inplace=True)
        this_input_table.drop(ATTRIBUTES_TO_OVERWRITE, axis=1, inplace=True)

        this_output_table = storm_object_table.loc[
            storm_object_table[FILE_INDEX_COLUMN] == i][
                OUTPUT_COLUMNS_FROM_BEST_TRACK]
        this_output_table = this_output_table.merge(
            this_input_table, on=COLUMNS_TO_MERGE_ON, how='left')
        tracking_io.write_processed_file(
            this_output_table, output_file_names[i])

    print '\n'


def write_simple_output_for_thea(storm_object_table, csv_file_name):
    """Writes output storm objects to CSV file for Thea.

    :param storm_object_table: pandas DataFrame with columns specified by
        `storm_tracking_io.write_processed_file`.
    :param csv_file_name: Path to output file.
    """

    error_checking.assert_columns_in_dataframe(
        storm_object_table, tracking_io.MANDATORY_COLUMNS)
    storm_object_table.sort_values(
        [tracking_utils.STORM_ID_COLUMN, tracking_utils.TIME_COLUMN], axis=0,
        ascending=[True, True], inplace=True)

    file_system_utils.mkdir_recursive_if_necessary(file_name=csv_file_name)
    csv_file_handle = open(csv_file_name, 'w')
    for j in range(len(OUTPUT_COLUMNS_FOR_THEA)):
        if j != 0:
            csv_file_handle.write(',')
        csv_file_handle.write('{0:s}'.format(OUTPUT_COLUMNS_FOR_THEA[j]))

    num_storm_objects = len(storm_object_table.index)
    for i in range(num_storm_objects):
        these_storm_indices = numpy.where(
            storm_object_table[tracking_utils.STORM_ID_COLUMN].values ==
            storm_object_table[tracking_utils.STORM_ID_COLUMN].values[i])[0]
        if len(these_storm_indices) < MIN_OBJECTS_IN_TRACK_FOR_THEA:
            continue

        csv_file_handle.write('\n')
        this_polygon_object = storm_object_table[
            tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN].values[i]

        for j in range(len(OUTPUT_COLUMNS_FOR_THEA)):
            if j != 0:
                csv_file_handle.write(',')

            if OUTPUT_COLUMNS_FOR_THEA[j] == VERTEX_LATITUDES_COLUMN:
                these_vertex_latitudes_deg = numpy.asarray(
                    this_polygon_object.exterior.xy[1])

                for k in range(len(these_vertex_latitudes_deg)):
                    if k != 0:
                        csv_file_handle.write(';')
                    csv_file_handle.write('{0:.5f}'.format(
                        these_vertex_latitudes_deg[k]))

            elif OUTPUT_COLUMNS_FOR_THEA[j] == VERTEX_LONGITUDES_COLUMN:
                these_vertex_longitudes_deg = numpy.asarray(
                    this_polygon_object.exterior.xy[0])

                for k in range(len(these_vertex_longitudes_deg)):
                    if k != 0:
                        csv_file_handle.write(';')
                    csv_file_handle.write('{0:.5f}'.format(
                        these_vertex_longitudes_deg[k]))

            elif OUTPUT_COLUMNS_FOR_THEA[j] == tracking_utils.STORM_ID_COLUMN:
                csv_file_handle.write('{0:s}'.format(
                    storm_object_table[
                        tracking_utils.STORM_ID_COLUMN].values[i]))

            elif OUTPUT_COLUMNS_FOR_THEA[j] == tracking_utils.TIME_COLUMN:
                csv_file_handle.write('{0:d}'.format(
                    storm_object_table[tracking_utils.TIME_COLUMN].values[i]))

            else:
                csv_file_handle.write('{0:.6f}'.format(
                    storm_object_table[OUTPUT_COLUMNS_FOR_THEA[j]].values[i]))

    csv_file_handle.close()
