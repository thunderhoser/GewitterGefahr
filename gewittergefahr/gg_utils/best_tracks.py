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
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): if this ends up being too slow, I can probably use
# `TheilSenRegressor` params to speed up the Theil-Sen fits.

EMPTY_STORM_ID = 'no_storm'

DEFAULT_MAX_EXTRAP_TIME_SEC = 610
DEFAULT_MAX_PREDICTION_ERROR_METRES = 10000.
DEFAULT_MAX_JOIN_TIME_SEC = 915
DEFAULT_MAX_JOIN_DISTANCE_METRES = 30000.
DEFAULT_MAX_MEAN_JOIN_ERROR_METRES = 10000.

DEFAULT_NUM_MAIN_ITERS = 3
DEFAULT_NUM_BREAKUP_ITERS = 5
DEFAULT_MIN_OBJECTS_IN_TRACK = 3

CENTROID_X_COLUMN = 'centroid_x_metres'
CENTROID_Y_COLUMN = 'centroid_y_metres'

TRACK_TIMES_COLUMN = 'unix_times_sec'
TRACK_START_TIME_COLUMN = 'start_time_unix_sec'
TRACK_END_TIME_COLUMN = 'end_time_unix_sec'
TRACK_X_COORDS_COLUMN = 'x_coords_metres'
TRACK_Y_COORDS_COLUMN = 'y_coords_metres'
OBJECT_INDICES_COLUMN_FOR_TRACK = 'object_indices'

THEIL_SEN_MODEL_X_COLUMN = 'theil_sen_model_for_x'
THEIL_SEN_MODEL_Y_COLUMN = 'theil_sen_model_for_y'
THEIL_SEN_MODEL_COLUMNS = [THEIL_SEN_MODEL_X_COLUMN, THEIL_SEN_MODEL_Y_COLUMN]

REPORT_PERIOD_FOR_THEIL_SEN = 100  # Print message after every 100 fits.
REPORT_PERIOD_FOR_BREAKUP = 100
REPORT_PERIOD_FOR_MERGER = 10
REPORT_PERIOD_FOR_TIE_BREAKER = 500

FILE_INDEX_COLUMN = 'file_index'
ORIG_STORM_ID_COLUMN = 'original_storm_id'

INPUT_COLUMNS_TO_KEEP = [
    tracking_io.STORM_ID_COLUMN, tracking_io.TIME_COLUMN,
    tracking_io.CENTROID_LAT_COLUMN, tracking_io.CENTROID_LNG_COLUMN]
COLUMNS_TO_MERGE_ON = [ORIG_STORM_ID_COLUMN, tracking_io.TIME_COLUMN]

EMPTY_TRACK_AGE_SEC = -1
OUTPUT_COLUMNS_TO_KEEP = [
    tracking_io.STORM_ID_COLUMN, tracking_io.TIME_COLUMN,
    tracking_io.AGE_COLUMN, tracking_io.TRACKING_START_TIME_COLUMN,
    tracking_io.TRACKING_END_TIME_COLUMN]


def _theil_sen_fit(
        unix_times_sec=None, x_coords_metres=None, y_coords_metres=None):
    """Fits Theil-Sen trajectory to storm track.

    N = number of storm objects in track

    :param unix_times_sec: length-N numpy array of times.
    :param x_coords_metres: length-N numpy array of x-coordinates.
    :param y_coords_metres: length-N numpy array of y-coordinates.
    :return: theil_sen_model_for_x: Instance of
        `sklearn.linear_model.TheilSenRegressor`, where the predictor is time
        and predictand is x-coordinate.
    :return: theil_sen_model_for_y: Instance of
        `sklearn.linear_model.TheilSenRegressor`, where the predictor is time
        and predictand is y-coordinate.
    """

    num_storm_objects = len(unix_times_sec)
    unix_times_sec = numpy.reshape(unix_times_sec, (num_storm_objects, 1))

    theil_sen_model_for_x = TheilSenRegressor(fit_intercept=True)
    theil_sen_model_for_x.fit(unix_times_sec, x_coords_metres)
    theil_sen_model_for_y = TheilSenRegressor(fit_intercept=True)
    theil_sen_model_for_y.fit(unix_times_sec, y_coords_metres)

    return theil_sen_model_for_x, theil_sen_model_for_y


def _theil_sen_predict(theil_sen_model_for_x=None, theil_sen_model_for_y=None,
                       query_time_unix_sec=None):
    """Uses Theil-Sen model to predict location of storm at given time.

    :param theil_sen_model_for_x: Instance of
        `sklearn.linear_model.TheilSenRegressor`, where the predictor is time
        and predictand is x-coordinate.
    :param theil_sen_model_for_y: Instance of
        `sklearn.linear_model.TheilSenRegressor`, where the predictor is time
        and predictand is y-coordinate.
    :param query_time_unix_sec: Query time (for which location will be
        predicted).
    :return: x_predicted_metres: Predicted x-coordinate.
    :return: y_predicted_metres: Predicted y-coordinate.
    """

    x_predicted_metres = theil_sen_model_for_x.predict(
        query_time_unix_sec)[0]
    y_predicted_metres = theil_sen_model_for_y.predict(
        query_time_unix_sec)[0]

    return x_predicted_metres, y_predicted_metres


def _get_prediction_errors_for_one_object(
        x_coord_metres=None, y_coord_metres=None, unix_time_sec=None,
        storm_track_table=None):
    """Computes Theil-Sen prediction errors for one storm object.

    N = number of tracks

    :param x_coord_metres: Actual x-coordinate of storm object.
    :param y_coord_metres: Actual y-coordinate of storm object.
    :param unix_time_sec: Valid time of storm object.
    :param storm_track_table: pandas DataFrame with columns documented in
        _theil_sen_fit_for_each_track.
    :return: prediction_errors_metres: length-N numpy array of prediction errors
        (distances between actual location and Theil-Sen prediction).
    """

    num_storm_tracks = len(storm_track_table)
    x_predicted_metres = numpy.full(num_storm_tracks, numpy.nan)
    y_predicted_metres = numpy.full(num_storm_tracks, numpy.nan)

    for j in range(num_storm_tracks):
        x_predicted_metres[j], y_predicted_metres[j] = _theil_sen_predict(
            theil_sen_model_for_x=storm_track_table[
                THEIL_SEN_MODEL_X_COLUMN].values[j],
            theil_sen_model_for_y=storm_track_table[
                THEIL_SEN_MODEL_Y_COLUMN].values[j],
            query_time_unix_sec=unix_time_sec)

    return numpy.sqrt(
        (x_coord_metres - x_predicted_metres) ** 2 +
        (y_coord_metres - y_predicted_metres) ** 2)


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


def _get_velocity_diff_for_two_tracks(theil_sen_models_for_x,
                                      theil_sen_models_for_y):
    """Computes velocity difference between two storm tracks.

    :param theil_sen_models_for_x: length-2 list.  Each element is an instance
        of `sklearn.linear_model.TheilSenRegressor` for one track, where the
        predictor is time and predictand is x-coordinate.
    :param theil_sen_models_for_y: Same as above, but for y-coordinates.
    :return: velocity_difference_m_s01: Magnitude of vectorial difference
        between the two velocities.
    """

    x_velocity_diff_m_s01 = (
        theil_sen_models_for_x[0].coef_[0] - theil_sen_models_for_x[1].coef_[0])
    y_velocity_diff_m_s01 = (
        theil_sen_models_for_y[0].coef_[0] - theil_sen_models_for_y[1].coef_[0])
    return numpy.sqrt(x_velocity_diff_m_s01 ** 2 + y_velocity_diff_m_s01 ** 2)


def _get_mean_prediction_error_for_two_tracks(
        x_coords_late_metres=None, y_coords_late_metres=None,
        late_times_unix_sec=None, theil_sen_model_for_x_early=None,
        theil_sen_model_for_y_early=None):
    """Computes mean Theil-Sen prediction error for two tracks.

    Specifically, for each time step t_L in the late track, the Theil-Sen model
    for the early track is used to predict the late track's position.  Thus, for
    each t_L, there is a prediction error (distance between actual and predicted
    locations).  This method returns the mean of said prediction errors.

    For the definitions of "early track" and "late track," see documentation for
    _get_join_time_for_two_tracks.

    T = number of times in late track

    :param x_coords_late_metres: length-T numpy array with x-coordinates of late
        track.
    :param y_coords_late_metres: length-T numpy array with y-coordinates of late
        track.
    :param late_times_unix_sec: length-T numpy array with times of late track.
    :param theil_sen_model_for_x_early: Instance of
        `sklearn.linear_model.TheilSenRegressor` for the early track, where the
        predictor is time and predictand is x-coordinate.
    :param theil_sen_model_for_y_early: Same as above, but for y-coordinates.
    :return: mean_prediction_error_metres: Mean prediction error for early track
        predicting positions in late track.
    """

    num_late_objects = len(x_coords_late_metres)
    x_predicted_metres = numpy.full(num_late_objects, numpy.nan)
    y_predicted_metres = numpy.full(num_late_objects, numpy.nan)

    for i in range(num_late_objects):
        x_predicted_metres[i], y_predicted_metres[i] = _theil_sen_predict(
            theil_sen_model_for_x=theil_sen_model_for_x_early,
            theil_sen_model_for_y=theil_sen_model_for_y_early,
            query_time_unix_sec=late_times_unix_sec[i])

    return numpy.mean(numpy.sqrt(
        (x_predicted_metres - x_coords_late_metres) ** 2 +
        (y_predicted_metres - y_coords_late_metres) ** 2))


def _break_ties_one_storm_track(
        object_x_coords_metres=None, object_y_coords_metres=None,
        object_times_unix_sec=None, theil_sen_model_for_x=None,
        theil_sen_model_for_y=None):
    """Breaks all ties for one storm track.

    For the definition of a "tie" among storm objects, see documentation for
    break_ties_among_storm_objects.

    T = number of times in track

    :param object_x_coords_metres: length-T numpy array with x-coordinates of
        track.
    :param object_y_coords_metres: length-T numpy array with y-coordinates of
        track.
    :param object_times_unix_sec: length-T numpy array with times of track.
    :param theil_sen_model_for_x: Instance of
        `sklearn.linear_model.TheilSenRegressor` for the track, where the
        predictor is time and predictand is x-coordinate.
    :param theil_sen_model_for_y: Same as above, except for y-coordinates.
    :return: indices_to_remove: 1-D numpy array with indices of storm objects to
        remove from the track.
    """

    num_storm_objects = len(object_times_unix_sec)
    object_indices_to_keep = numpy.linspace(
        0, num_storm_objects - 1, num=num_storm_objects, dtype=int)

    unique_times_unix_sec, orig_to_unique_time_indices = numpy.unique(
        object_times_unix_sec, return_inverse=True)
    num_unique_times = len(unique_times_unix_sec)

    while num_unique_times < len(object_times_unix_sec):
        for i in range(num_unique_times):
            these_object_indices = numpy.where(
                orig_to_unique_time_indices == i)[0]
            if len(these_object_indices) == 1:
                continue

            this_x_predicted_metres, this_y_predicted_metres = (
                _theil_sen_predict(
                    theil_sen_model_for_x=theil_sen_model_for_x,
                    theil_sen_model_for_y=theil_sen_model_for_y,
                    query_time_unix_sec=unique_times_unix_sec[i]))

            these_prediction_errors_metres = numpy.sqrt(
                (object_x_coords_metres[these_object_indices] -
                 this_x_predicted_metres) ** 2 +
                (object_y_coords_metres[these_object_indices] -
                 this_y_predicted_metres) ** 2)
            this_min_index = these_object_indices[
                numpy.argmin(these_prediction_errors_metres)]

            for k in these_object_indices:
                if k == this_min_index:
                    continue

                object_x_coords_metres = numpy.delete(object_x_coords_metres, k)
                object_y_coords_metres = numpy.delete(object_y_coords_metres, k)
                object_times_unix_sec = numpy.delete(object_times_unix_sec, k)
                object_indices_to_keep = numpy.delete(object_indices_to_keep, k)

            theil_sen_model_for_x, theil_sen_model_for_y = _theil_sen_fit(
                unix_times_sec=object_times_unix_sec,
                x_coords_metres=object_x_coords_metres,
                y_coords_metres=object_y_coords_metres)

            break

        unique_times_unix_sec, orig_to_unique_time_indices = numpy.unique(
            object_times_unix_sec, return_inverse=True)
        num_unique_times = len(unique_times_unix_sec)

    remove_object_flags = numpy.full(num_storm_objects, True, dtype=bool)
    remove_object_flags[object_indices_to_keep] = False
    return numpy.where(remove_object_flags)[0]


def _find_changed_tracks(storm_track_table, orig_storm_track_table):
    """Finds tracks that chgd from orig_storm_track_table to storm_track_table.

    :param storm_track_table: pandas DataFrame with columns documented in
        _theil_sen_fit_for_each_track.
    :param orig_storm_track_table: pandas DataFrame with columns documented in
        _theil_sen_fit_for_each_track.
    :return: track_changed_indices: 1-D numpy array of rows (indices into
        storm_track_table) for which track changed.
    """

    num_storm_tracks = len(storm_track_table.index)
    orig_storm_ids = numpy.asarray(
        orig_storm_track_table[tracking_io.STORM_ID_COLUMN].values)
    track_changed_flags = numpy.full(num_storm_tracks, False, dtype=bool)

    for j in range(num_storm_tracks):
        this_storm_in_orig_flags = (
            orig_storm_ids == storm_track_table[
                tracking_io.STORM_ID_COLUMN].values[j])
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
        max_join_time_sec=DEFAULT_MAX_JOIN_TIME_SEC,
        max_join_distance_metres=DEFAULT_MAX_JOIN_DISTANCE_METRES,
        max_mean_join_error_metres=DEFAULT_MAX_MEAN_JOIN_ERROR_METRES,
        max_velocity_diff_for_join_m_s01=None,
        num_main_iters=DEFAULT_NUM_MAIN_ITERS,
        num_breakup_iters=DEFAULT_NUM_BREAKUP_ITERS,
        min_objects_in_track=DEFAULT_MIN_OBJECTS_IN_TRACK):
    """Checks best-track parameters for errors.

    :param max_extrap_time_for_breakup_sec: See documentation for
        run_best_track.
    :param max_prediction_error_for_breakup_metres: See doc for run_best_track.
    :param max_join_time_sec: See doc for run_best_track.
    :param max_join_distance_metres: See doc for run_best_track.
    :param max_mean_join_error_metres: See doc for run_best_track.
    :param max_velocity_diff_for_join_m_s01: See doc for run_best_track.
    :param num_main_iters: See doc for run_best_track.
    :param num_breakup_iters: See doc for run_best_track.
    :param min_objects_in_track: See doc for run_best_track.
    """

    error_checking.assert_is_integer(max_extrap_time_for_breakup_sec)
    error_checking.assert_is_greater(max_extrap_time_for_breakup_sec, 0)
    error_checking.assert_is_greater(
        max_prediction_error_for_breakup_metres, 0.)
    error_checking.assert_is_integer(max_join_time_sec)
    error_checking.assert_is_greater(max_join_time_sec, 0)
    error_checking.assert_is_greater(max_join_distance_metres, 0.)
    error_checking.assert_is_greater(max_mean_join_error_metres, 0.)

    if max_velocity_diff_for_join_m_s01 is not None:
        error_checking.assert_is_greater(max_velocity_diff_for_join_m_s01, 0.)

    error_checking.assert_is_integer(num_main_iters)
    error_checking.assert_is_greater(num_main_iters, 0)
    error_checking.assert_is_integer(num_breakup_iters)
    error_checking.assert_is_greater(num_breakup_iters, 0)
    error_checking.assert_is_integer(min_objects_in_track)
    error_checking.assert_is_geq(min_objects_in_track, 1)


def storm_objects_to_tracks(storm_object_table, storm_ids_to_use=None):
    """Adds tracking info to storm objects.

    T = number of times (storm objects) in a given track

    :param storm_object_table: pandas DataFrame with at least the following
        columns.  Each row is one storm object.
    storm_object_table.storm_id: String ID.
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
        storm_object_table[tracking_io.STORM_ID_COLUMN].values)
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

    storm_track_dict = {tracking_io.STORM_ID_COLUMN: storm_ids_to_use}
    storm_track_table = pandas.DataFrame.from_dict(storm_track_dict)

    num_storms_to_use = len(storm_ids_to_use)
    simple_array = numpy.full(num_storms_to_use, numpy.nan, dtype=int)
    nested_array = storm_track_table[[
        tracking_io.STORM_ID_COLUMN,
        tracking_io.STORM_ID_COLUMN]].values.tolist()

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
            storm_object_table[tracking_io.TIME_COLUMN].values[
                these_storm_object_indices])
        these_storm_object_indices = these_storm_object_indices[sort_indices]

        this_table_index = storm_ids_to_use.index(unique_storm_ids[i])

        storm_track_table[TRACK_TIMES_COLUMN].values[this_table_index] = (
            storm_object_table[tracking_io.TIME_COLUMN].values[
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


def theil_sen_fit_for_each_track(
        storm_track_table, fit_indices=None, verbose=True):
    """Fits Theil-Sen model for each storm track.

    :param storm_track_table: pandas DataFrame created by
        storm_objects_to_tracks.
    :param fit_indices: 1-D numpy array of indices (rows in storm_track_table).
        Theil-Sen fits will be computed only for these storm tracks.  If
        fit_indices = None, Theil-Sen fits will be computed for all storm
        tracks.
    :param verbose: Boolean flag.  If verbose = True, will print progress
        messages to the command window.  If verbose = False, will print nothing.
    :return: storm_track_table: Same as input, but with additional columns
        listed below.
    storm_track_table.theil_sen_model_for_x: Instance of
        `sklearn.linear_model.TheilSenRegressor`, where the predictor is time
        and predictand is x-coordinate.
    storm_track_table.theil_sen_model_for_y: Instance of
        `sklearn.linear_model.TheilSenRegressor`, where the predictor is time
        and predictand is y-coordinate.
    """

    num_storm_tracks = len(storm_track_table.index)
    if fit_indices is None:
        fit_indices = numpy.linspace(
            0, num_storm_tracks - 1, num=num_storm_tracks, dtype=int)

    error_checking.assert_is_integer_numpy_array(fit_indices)
    error_checking.assert_is_numpy_array(fit_indices, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(fit_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        fit_indices, num_storm_tracks)

    if not (THEIL_SEN_MODEL_X_COLUMN in list(storm_track_table) and
            THEIL_SEN_MODEL_Y_COLUMN in list(storm_track_table)):
        object_array = numpy.full(num_storm_tracks, numpy.nan, dtype=object)
        argument_dict = {THEIL_SEN_MODEL_X_COLUMN: object_array,
                         THEIL_SEN_MODEL_Y_COLUMN: object_array}
        storm_track_table = storm_track_table.assign(**argument_dict)

    num_fits_computed = 0
    for i in fit_indices:
        (storm_track_table[THEIL_SEN_MODEL_X_COLUMN].values[i],
         storm_track_table[THEIL_SEN_MODEL_Y_COLUMN].values[i]) = (
             _theil_sen_fit(
                 unix_times_sec=storm_track_table[TRACK_TIMES_COLUMN].values[i],
                 x_coords_metres=storm_track_table[
                     TRACK_X_COORDS_COLUMN].values[i],
                 y_coords_metres=storm_track_table[
                     TRACK_Y_COORDS_COLUMN].values[i]))

        num_fits_computed += 1
        if not (numpy.mod(num_fits_computed, REPORT_PERIOD_FOR_THEIL_SEN) == 0
                and verbose):
            continue

        print ('Have fit Theil-Sen model for ' + str(num_fits_computed) + '/' +
               str(len(fit_indices)) + ' storm tracks...')

    if verbose:
        print ('Have fit Theil-Sen model for all ' + str(len(fit_indices)) +
               ' storm tracks!')
    return storm_track_table


def break_storm_tracks(
        storm_object_table=None, storm_track_table=None,
        working_object_indices=None,
        max_extrapolation_time_sec=DEFAULT_MAX_EXTRAP_TIME_SEC,
        max_prediction_error_metres=DEFAULT_MAX_PREDICTION_ERROR_METRES):
    """Breaks storm tracks and reassigns storm objects.

    This is the "break-up" step in w2besttrack.

    :param storm_object_table: pandas DataFrame with at least the following
        columns.  Each row is one storm object.
    storm_object_table.storm_id: String ID.
    storm_object_table.unix_time_sec: Valid time.
    storm_object_table.centroid_x_metres: x-coordinate of storm centroid.
    storm_object_table.centroid_y_metres: y-coordinate of storm centroid.

    :param storm_track_table: pandas DataFrame with columns documented in
        theil_sen_fit_for_each_track.
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
    :return: storm_object_table: Same as input, except that "storm_id" column
        may have changed for several objects.
    :return: storm_track_table: Same as input, except that values may have
        changed for several tracks.
    """

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
        storm_object_table[tracking_io.STORM_ID_COLUMN].values[
            working_object_indices])

    for i in working_object_indices:
        if numpy.mod(num_objects_done, REPORT_PERIOD_FOR_BREAKUP) == 0:
            print ('Have performed break-up step for ' + str(num_objects_done) +
                   '/' + str(num_working_objects) + ' storm objects...')

        num_objects_done += 1

        times_before_start_sec = (
            storm_track_table[TRACK_START_TIME_COLUMN].values -
            storm_object_table[tracking_io.TIME_COLUMN].values[i])
        times_after_end_sec = (
            storm_object_table[tracking_io.TIME_COLUMN].values[i] -
            storm_track_table[TRACK_END_TIME_COLUMN].values)

        try_track_flags = numpy.logical_and(
            times_before_start_sec <= max_extrapolation_time_sec,
            times_after_end_sec <= max_extrapolation_time_sec)
        if not numpy.any(try_track_flags):
            continue

        try_track_indices = numpy.where(try_track_flags)[0]
        prediction_errors_metres = _get_prediction_errors_for_one_object(
            x_coord_metres=storm_object_table[CENTROID_X_COLUMN].values[i],
            y_coord_metres=storm_object_table[CENTROID_Y_COLUMN].values[i],
            unix_time_sec=storm_object_table[tracking_io.TIME_COLUMN].values[i],
            storm_track_table=storm_track_table.iloc[try_track_indices])

        if numpy.min(prediction_errors_metres) > max_prediction_error_metres:
            continue

        nearest_track_index = try_track_indices[
            numpy.argmin(prediction_errors_metres)]
        storm_object_table[tracking_io.STORM_ID_COLUMN].values[i] = (
            storm_track_table[tracking_io.STORM_ID_COLUMN].values[
                nearest_track_index])

    print ('Have performed break-up step for all ' + str(num_working_objects) +
           ' storm objects!')

    new_storm_ids = numpy.asarray(
        storm_object_table[tracking_io.STORM_ID_COLUMN].values[
            working_object_indices])
    object_changed_indices = numpy.where(orig_storm_ids != new_storm_ids)[0]
    print ('Assigned ' + str(len(object_changed_indices)) +
           ' storm objects to a new track during this procedure.\n\n')

    orig_storm_track_table = copy.deepcopy(storm_track_table)
    storm_track_table = storm_objects_to_tracks(storm_object_table)
    storm_track_table = storm_track_table.merge(
        orig_storm_track_table[
            THEIL_SEN_MODEL_COLUMNS + [tracking_io.STORM_ID_COLUMN]],
        on=tracking_io.STORM_ID_COLUMN, how='left')

    track_changed_indices = _find_changed_tracks(
        storm_track_table, orig_storm_track_table)
    storm_track_table = theil_sen_fit_for_each_track(
        storm_track_table, fit_indices=track_changed_indices, verbose=False)
    return storm_object_table, storm_track_table


def merge_storm_tracks(
        storm_object_table=None, storm_track_table=None,
        working_track_indices=None, max_join_time_sec=DEFAULT_MAX_JOIN_TIME_SEC,
        max_join_distance_metres=DEFAULT_MAX_JOIN_DISTANCE_METRES,
        max_mean_prediction_error_metres=DEFAULT_MAX_MEAN_JOIN_ERROR_METRES,
        max_velocity_diff_m_s01=None):
    """Merges pairs of similar storm tracks.

    This is the "merger" step in w2besttrack.

    :param storm_object_table: See documentation for break_storm_tracks.
    :param storm_track_table: pandas DataFrame with columns documented in
        theil_sen_fit_for_each_track.
    :param working_track_indices: 1-D numpy array with indices of storm tracks
        to work on (consider for merging).  If working_track_indices = None,
        this method will work on all storm tracks.
    :param max_join_time_sec: Maximum time between tracks (specifically, between
        end time of early track and start time of late track).
    :param max_join_distance_metres: Maximum distance between tracks
        (specifically, between last position of early track and first position
        of late track).
    :param max_mean_prediction_error_metres: Maximum mean error for early track
        predicting positions in late track.  For more explanation, see
        _get_mean_prediction_error_for_two_tracks.
    :param max_velocity_diff_m_s01: Maximum difference (metres per second)
        between Theil-Sen velocities of the two tracks.  If None, this
        constraint will not be enforced.
    :return: storm_object_table: Same as input, except that "storm_id" column
        may have changed for several objects.
    :return: storm_track_table: Same as input, except that values may have
        changed for several tracks.
    """

    check_best_track_params(
        max_join_time_sec=max_join_time_sec,
        max_join_distance_metres=max_join_distance_metres,
        max_mean_join_error_metres=max_mean_prediction_error_metres,
        max_velocity_diff_for_join_m_s01=max_velocity_diff_m_s01)

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
        if numpy.mod(num_tracks_considered, REPORT_PERIOD_FOR_MERGER) == 0:
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

            if this_join_distance_metres > max_join_distance_metres:
                continue

            if max_velocity_diff_m_s01 is None:
                this_velocity_diff_m_s01 = _get_velocity_diff_for_two_tracks(
                    storm_track_table[THEIL_SEN_MODEL_X_COLUMN].values[
                        these_track_indices],
                    storm_track_table[THEIL_SEN_MODEL_Y_COLUMN].values[
                        these_track_indices])

                if this_velocity_diff_m_s01 > max_velocity_diff_m_s01:
                    continue

            this_mean_prediction_error_metres = (
                _get_mean_prediction_error_for_two_tracks(
                    x_coords_late_metres=
                    storm_track_table[TRACK_X_COORDS_COLUMN].values[late_index],
                    y_coords_late_metres=
                    storm_track_table[TRACK_Y_COORDS_COLUMN].values[late_index],
                    late_times_unix_sec=
                    storm_track_table[TRACK_TIMES_COLUMN].values[late_index],
                    theil_sen_model_for_x_early=storm_track_table[
                        THEIL_SEN_MODEL_X_COLUMN].values[early_index],
                    theil_sen_model_for_y_early=storm_track_table[
                        THEIL_SEN_MODEL_Y_COLUMN].values[early_index]))

            if (this_mean_prediction_error_metres >
                    max_mean_prediction_error_metres):
                continue

            remove_storm_track_flags[k] = True
            num_pairs_merged += 1

            storm_id_j = storm_track_table[
                tracking_io.STORM_ID_COLUMN].values[j]
            storm_id_k = storm_track_table[
                tracking_io.STORM_ID_COLUMN].values[k]

            for i in range(len(storm_object_table.index)):
                if storm_object_table[
                        tracking_io.STORM_ID_COLUMN].values[i] == storm_id_k:
                    storm_object_table[
                        tracking_io.STORM_ID_COLUMN].values[i] = storm_id_j

            storm_track_table_j_only = storm_objects_to_tracks(
                storm_object_table, [storm_id_j])
            storm_track_table_j_only = theil_sen_fit_for_each_track(
                storm_track_table_j_only, verbose=False)
            storm_track_table.iloc[j] = storm_track_table_j_only.iloc[0]

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
    :param storm_track_table: pandas DataFrame with columns documented in
        theil_sen_fit_for_each_track.
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
        if numpy.mod(num_tracks_considered, REPORT_PERIOD_FOR_TIE_BREAKER) == 0:
            print ('Have considered ' + str(num_tracks_considered) + '/' +
                   str(num_working_tracks) +
                   ' storm tracks for tie-breaking...')

        num_tracks_considered += 1

        these_object_indices_to_remove = _break_ties_one_storm_track(
            object_x_coords_metres=
            storm_track_table[TRACK_X_COORDS_COLUMN].values[j],
            object_y_coords_metres=
            storm_track_table[TRACK_Y_COORDS_COLUMN].values[j],
            object_times_unix_sec=
            storm_track_table[TRACK_TIMES_COLUMN].values[j],
            theil_sen_model_for_x=
            storm_track_table[THEIL_SEN_MODEL_X_COLUMN].values[j],
            theil_sen_model_for_y=
            storm_track_table[THEIL_SEN_MODEL_Y_COLUMN].values[j])

        these_object_indices_to_remove = storm_track_table[
            OBJECT_INDICES_COLUMN_FOR_TRACK].values[j][
                these_object_indices_to_remove]
        if not len(these_object_indices_to_remove):
            continue

        num_ties_broken += len(these_object_indices_to_remove)
        num_tracks_with_ties_broken += 1
        for i in these_object_indices_to_remove:
            storm_object_table[tracking_io.STORM_ID_COLUMN].values[
                i] = EMPTY_STORM_ID

        storm_id_j = storm_track_table[
            tracking_io.STORM_ID_COLUMN].values[j]
        storm_track_table_j_only = storm_objects_to_tracks(
            storm_object_table, [storm_id_j])
        storm_track_table_j_only = theil_sen_fit_for_each_track(
            storm_track_table_j_only, verbose=False)
        storm_track_table.iloc[j] = storm_track_table_j_only.iloc[0]

    print ('Have considered all ' + str(num_working_tracks) +
           ' storm tracks for tie-breaking!')
    print ('Broke ' + str(num_ties_broken) + ' ties in ' +
           str(num_tracks_with_ties_broken) +
           ' tracks during this procedure.\n\n')

    return storm_object_table, storm_track_table


def remove_short_tracks(
        storm_object_table, min_objects_in_track=DEFAULT_MIN_OBJECTS_IN_TRACK):
    """Removes storm tracks without enough storm objects.

    :param storm_object_table: pandas DataFrame with at least the following
        columns.  Each row is one storm object.
    storm_object_table.storm_id: String ID.
    storm_object_table.unix_time_sec: Valid time.

    :param min_objects_in_track: Minimum number of storm objects in a track.
    :return: storm_object_table: Same as input, except that all objects
        belonging to a track with < `min_objects_in_track` objects have been
        removed.
    """

    check_best_track_params(min_objects_in_track=min_objects_in_track)

    storm_id_by_object = numpy.asarray(
        storm_object_table[tracking_io.STORM_ID_COLUMN].values)
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


def recompute_attributes(
        storm_object_table, best_track_start_time_unix_sec=None,
        best_track_end_time_unix_sec=None):
    """Recomputes the following storm attributes, using new tracks.

    - age of storm track
    - start time of tracking period
    - end time of tracking period

    :param storm_object_table: pandas DataFrame with at least the following
        columns.  Each row is one storm object.
    storm_object_table.storm_id: String ID.
    storm_object_table.unix_time_sec: Valid time.

    :param best_track_start_time_unix_sec: Start of tracking period.
    :param best_track_end_time_unix_sec: End of tracking period.
    :return: storm_object_table: Same as input, but with additional columns
        listed below.
    storm_object_table.age_sec: Age of storm track.
    storm_object_table.tracking_start_time_unix_sec: Start of tracking period.
    storm_object_table.tracking_end_time_unix_sec: End of tracking period.
    """

    error_checking.assert_is_integer(best_track_start_time_unix_sec)
    error_checking.assert_is_integer(best_track_end_time_unix_sec)
    error_checking.assert_is_greater(
        best_track_end_time_unix_sec, best_track_start_time_unix_sec)

    num_storm_objects = len(storm_object_table.index)
    tracking_start_times_unix_sec = numpy.full(
        num_storm_objects, best_track_start_time_unix_sec, dtype=int)
    tracking_end_times_unix_sec = numpy.full(
        num_storm_objects, best_track_end_time_unix_sec, dtype=int)
    track_ages_sec = numpy.full(
        num_storm_objects, EMPTY_TRACK_AGE_SEC, dtype=int)

    storm_id_by_object = numpy.asarray(
        storm_object_table[tracking_io.STORM_ID_COLUMN].values)
    unique_storm_ids, storm_ids_object_to_unique = numpy.unique(
        storm_id_by_object, return_inverse=True)

    for i in range(len(unique_storm_ids)):
        these_object_indices = numpy.where(storm_ids_object_to_unique == i)[0]
        this_start_time_unix_sec = numpy.min(
            storm_object_table[tracking_io.TIME_COLUMN].values[
                these_object_indices])
        if this_start_time_unix_sec == best_track_start_time_unix_sec:
            continue

        track_ages_sec[these_object_indices] = (
            storm_object_table[tracking_io.TIME_COLUMN].values[
                these_object_indices] - this_start_time_unix_sec)

    argument_dict = {
        tracking_io.TRACKING_START_TIME_COLUMN: tracking_start_times_unix_sec,
        tracking_io.TRACKING_END_TIME_COLUMN: tracking_end_times_unix_sec,
        tracking_io.AGE_COLUMN: track_ages_sec
    }
    return storm_object_table.assign(**argument_dict)


def run_best_track(
        storm_object_table=None,
        max_extrap_time_for_breakup_sec=DEFAULT_MAX_EXTRAP_TIME_SEC,
        max_prediction_error_for_breakup_metres=
        DEFAULT_MAX_PREDICTION_ERROR_METRES,
        max_join_time_sec=DEFAULT_MAX_JOIN_TIME_SEC,
        max_join_distance_metres=DEFAULT_MAX_JOIN_DISTANCE_METRES,
        max_mean_join_error_metres=DEFAULT_MAX_MEAN_JOIN_ERROR_METRES,
        max_velocity_diff_for_join_m_s01=None,
        num_main_iters=DEFAULT_NUM_MAIN_ITERS,
        num_breakup_iters=DEFAULT_NUM_BREAKUP_ITERS,
        min_objects_in_track=DEFAULT_MIN_OBJECTS_IN_TRACK):
    """Runs the full w2besttrack algorithm without IO.

    In other words, original tracks must be read before this method and new
    tracks must be written after this method.

    :param storm_object_table: pandas DataFrame with at least the following
        columns.  Each row is one storm object.
    storm_object_table.storm_id: String ID for storm track.
    storm_object_table.unix_time_sec: Valid time for storm object.
    storm_object_table.centroid_lat_deg: Latitude (deg N) of storm-object
        centroid.
    storm_object_table.centroid_lng_deg: Longitude (deg E) of storm-object
        centroid.

    :param max_extrap_time_for_breakup_sec: See doc for break_storm_tracks.
    :param max_prediction_error_for_breakup_metres: See doc for
        break_storm_tracks.
    :param max_join_time_sec: See documentation for merge_storm_tracks.
    :param max_join_distance_metres: See documentation for merge_storm_tracks.
    :param max_mean_join_error_metres: See doc for merge_storm_tracks.
    :param max_velocity_diff_for_join_m_s01: See doc for merge_storm_tracks.
    :param num_main_iters: Number of main iterations (outer loops).  Each main
        iteration consists of `num_breakup_iters` break-up iterations, one
        merger iteration, and one tie-breaking iteration.
    :param num_breakup_iters: Number of break-up iterations (inner loops).
    :param min_objects_in_track: Minimum number of storm objects in a track.
        After all the main iterations, any track with < `min_objects_in_track`
        storm objects will be removed.
    :return: storm_object_table: Same as input, except that "storm_id" may have
        changed for several storm objects.  Also, this will be without the
        columns "centroid_lat_deg" and "centroid_lng_deg".
    """

    check_best_track_params(
        max_extrap_time_for_breakup_sec=max_extrap_time_for_breakup_sec,
        max_prediction_error_for_breakup_metres=
        max_prediction_error_for_breakup_metres,
        max_join_time_sec=max_join_time_sec,
        max_join_distance_metres=max_join_distance_metres,
        max_mean_join_error_metres=max_mean_join_error_metres,
        max_velocity_diff_for_join_m_s01=max_velocity_diff_for_join_m_s01,
        num_main_iters=num_main_iters, num_breakup_iters=num_breakup_iters,
        min_objects_in_track=min_objects_in_track)

    global_centroid_lat_deg, global_centroid_lng_deg = (
        polygons.get_latlng_centroid(
            storm_object_table[tracking_io.CENTROID_LAT_COLUMN].values,
            storm_object_table[tracking_io.CENTROID_LNG_COLUMN].values))
    projection_object = projections.init_azimuthal_equidistant_projection(
        global_centroid_lat_deg, global_centroid_lng_deg)
    x_centroids_metres, y_centroids_metres = projections.project_latlng_to_xy(
        storm_object_table[tracking_io.CENTROID_LAT_COLUMN].values,
        storm_object_table[tracking_io.CENTROID_LNG_COLUMN].values,
        projection_object=projection_object, false_easting_metres=0.,
        false_northing_metres=0.)

    argument_dict = {CENTROID_X_COLUMN: x_centroids_metres,
                     CENTROID_Y_COLUMN: y_centroids_metres}
    storm_object_table = storm_object_table.assign(**argument_dict)
    storm_object_table.drop(
        [tracking_io.CENTROID_LAT_COLUMN, tracking_io.CENTROID_LNG_COLUMN],
        axis=1, inplace=True)

    storm_track_table = storm_objects_to_tracks(storm_object_table)
    storm_track_table = theil_sen_fit_for_each_track(storm_track_table)

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
                max_prediction_error_for_breakup_metres)

        storm_object_table, storm_track_table = merge_storm_tracks(
            storm_object_table=storm_object_table,
            storm_track_table=storm_track_table,
            max_join_time_sec=max_join_time_sec,
            max_join_distance_metres=max_join_distance_metres,
            max_mean_prediction_error_metres=max_mean_join_error_metres,
            max_velocity_diff_m_s01=max_velocity_diff_for_join_m_s01)

        storm_object_table, storm_track_table = break_ties_among_storm_objects(
            storm_object_table, storm_track_table)

    best_track_start_time_unix_sec = numpy.min(
        storm_object_table[tracking_io.TIME_COLUMN].values)
    best_track_end_time_unix_sec = numpy.max(
        storm_object_table[tracking_io.TIME_COLUMN].values)

    print ('Removing storm tracks with < ' + str(min_objects_in_track) +
           ' objects...')
    storm_object_table = remove_short_tracks(
        storm_object_table, min_objects_in_track=min_objects_in_track)

    print 'Recomputing storm attributes...'
    return recompute_attributes(
        storm_object_table,
        best_track_start_time_unix_sec=best_track_start_time_unix_sec,
        best_track_end_time_unix_sec=best_track_end_time_unix_sec)


def read_input_storm_objects(input_file_names, keep_spc_date=False):
    """Reads input storm objects from one or more files.

    Input files should be in the format produced by
    `storm_tracking_io.write_processed_file`.

    :param input_file_names: 1-D list of paths to input files.
    :param keep_spc_date: Boolean flag.  If True, will keep the column
        "spc_date_unix_sec" in the input files.  If False, will throw it out.
    :return: storm_object_table: pandas DataFrame with the following columns.
        Each row is one storm object.
    storm_object_table.storm_id: String ID for storm track.
    storm_object_table.original_storm_id: Original string ID for storm track.
    storm_object_table.unix_time_sec: Valid time of storm object.
    storm_object_table.centroid_lat_deg: Latitude (deg N) of storm-object
        centroid.
    storm_object_table.centroid_lng_deg: Longitude (deg E) of storm-object
        centroid.
    storm_object_table.file_index: Index of file from which storm object was
        read.  This is an index into `storm_object_file_names`.
    """

    error_checking.assert_is_string_list(input_file_names)
    error_checking.assert_is_numpy_array(
        numpy.asarray(input_file_names), num_dimensions=1)

    error_checking.assert_is_boolean(keep_spc_date)
    if keep_spc_date:
        columns_to_keep = INPUT_COLUMNS_TO_KEEP + [tracking_io.SPC_DATE_COLUMN]
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
        ORIG_STORM_ID_COLUMN:
            storm_object_table[tracking_io.STORM_ID_COLUMN].values}
    return storm_object_table.assign(**argument_dict)


def write_output_storm_objects(
        storm_object_table, input_file_names=None, output_file_names=None):
    """Writes output storm objects (after best-track) to one or more files.

    N = number of input files = number of output files

    :param storm_object_table: pandas DataFrame with the following columns.
        Each row is one storm object.
    storm_object_table.storm_id: String ID for storm track.
    storm_object_table.original_storm_id: Original string ID for storm track.
    storm_object_table.unix_time_sec: Valid time of storm object.
    storm_object_table.file_index: Index of file from which storm object was
        read.  This is an index into `storm_object_file_names`.
    storm_object_table.age_sec: Age of storm track.
    storm_object_table.tracking_start_time_unix_sec: Start of tracking period.
    storm_object_table.tracking_end_time_unix_sec: End of tracking period.

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
            tracking_io.STORM_ID_COLUMN: ORIG_STORM_ID_COLUMN}
        this_input_table.rename(columns=column_dict_old_to_new, inplace=True)

        input_columns_to_drop = set(list(OUTPUT_COLUMNS_TO_KEEP))
        for this_column in COLUMNS_TO_MERGE_ON:
            if this_column in input_columns_to_drop:
                input_columns_to_drop.remove(this_column)

        input_columns_to_drop = list(input_columns_to_drop.intersection(
            set(list(this_input_table))))
        this_input_table.drop(input_columns_to_drop, axis=1, inplace=True)

        this_output_table = storm_object_table.loc[
            storm_object_table[FILE_INDEX_COLUMN] == i][OUTPUT_COLUMNS_TO_KEEP]
        this_output_table = this_output_table.merge(
            this_input_table, on=COLUMNS_TO_MERGE_ON, how='left')

        tracking_io.write_processed_file(
            this_output_table, output_file_names[i])

    print '\n'
