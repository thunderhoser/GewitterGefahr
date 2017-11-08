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

# TODO(thunderhoser): will the `pandas.merge` commands work without duplicating
# columns?

DEFAULT_MAX_EXTRAP_TIME_SEC = 610
DEFAULT_MAX_PREDICTION_ERROR_METRES = 10000.
DEFAULT_MAX_JOIN_TIME_SEC = 915
DEFAULT_MAX_JOIN_DISTANCE_METRES = 30000.
DEFAULT_MAX_MEAN_JOIN_ERROR_METRES = 10000.

CENTROID_X_COLUMN = 'centroid_x_metres'
CENTROID_Y_COLUMN = 'centroid_y_metres'

TRACK_TIMES_COLUMN = 'unix_times_sec'
TRACK_START_TIME_COLUMN = 'start_time_unix_sec'
TRACK_END_TIME_COLUMN = 'end_time_unix_sec'
TRACK_X_COORDS_COLUMN = 'x_coords_metres'
TRACK_Y_COORDS_COLUMN = 'y_coords_metres'
THEIL_SEN_MODEL_X_COLUMN = 'theil_sen_model_for_x'
THEIL_SEN_MODEL_Y_COLUMN = 'theil_sen_model_for_y'

REPORT_PERIOD_FOR_THEIL_SEN = 100  # Print message after every 100 fits.


def _theil_sen_fit(unix_times_sec=None, x_coords_metres=None,
                   y_coords_metres=None):
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

    # TODO(thunderhoser): what does this do with only one data point?

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


def _storm_objects_to_tracks(storm_object_table, storm_ids_to_use=None):
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
    storm_track_table.x_coords_metres: length-T numpy array of x-coordinates.
    storm_track_table.y_coords_metres: length-T numpy array of y-coordinates.
    """

    storm_id_by_object = numpy.asarray(
        storm_object_table[tracking_io.STORM_ID_COLUMN].values)
    unique_storm_ids, storm_ids_object_to_unique = numpy.unique(
        storm_id_by_object, return_inverse=True)

    if storm_ids_to_use is None:
        storm_ids_to_use = unique_storm_ids.tolist()
    storm_track_dict = {tracking_io.STORM_ID_COLUMN: storm_ids_to_use}
    storm_track_table = pandas.DataFrame.from_dict(storm_track_dict)

    num_storms_to_use = len(storm_ids_to_use)
    simple_array = numpy.full(num_storms_to_use, numpy.nan, dtype=int)
    nested_array = storm_object_table[[
        tracking_io.STORM_ID_COLUMN,
        tracking_io.STORM_ID_COLUMN]].values.tolist()

    argument_dict = {TRACK_TIMES_COLUMN: nested_array,
                     TRACK_X_COORDS_COLUMN: nested_array,
                     TRACK_Y_COORDS_COLUMN: nested_array,
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

        storm_track_table[TRACK_START_TIME_COLUMN].values[this_table_index] = (
            storm_track_table[TRACK_TIMES_COLUMN].values[this_table_index][0])
        storm_track_table[TRACK_END_TIME_COLUMN].values[this_table_index] = (
            storm_track_table[TRACK_TIMES_COLUMN].values[this_table_index][-1])

    return storm_track_table


def _find_changed_tracks(storm_track_table, orig_storm_track_table):
    """Finds tracks that chgd from orig_storm_track_table to storm_track_table.

    :param storm_track_table: pandas DataFrame created by
        _storm_objects_to_tracks.
    :param orig_storm_track_table: pandas DataFrame created by
        _storm_objects_to_tracks.
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


def _theil_sen_fit_for_each_track(storm_track_table, fit_indices=None):
    """Fits Theil-Sen model for each storm track.

    :param storm_track_table: pandas DataFrame created by
        _storm_objects_to_tracks.
    :param fit_indices: 1-D numpy array of indices (rows in storm_track_table).
        Theil-Sen fits will be computed only for these storm tracks.  If
        fit_indices = None, Theil-Sen fits will be computed for all storm
        tracks.
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

    if not (THEIL_SEN_MODEL_X_COLUMN in list(storm_track_table) and
            THEIL_SEN_MODEL_Y_COLUMN in list(storm_track_table)):
        object_array = numpy.full(num_storm_tracks, numpy.nan, dtype=object)
        argument_dict = {THEIL_SEN_MODEL_X_COLUMN: object_array,
                         THEIL_SEN_MODEL_Y_COLUMN: object_array}
        storm_track_table = storm_track_table.assign(**argument_dict)

    if fit_indices is None:
        fit_indices = numpy.linspace(
            0, num_storm_tracks - 1, num=num_storm_tracks, dtype=int)

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
        if numpy.mod(num_fits_computed, REPORT_PERIOD_FOR_THEIL_SEN) != 0:
            continue

        print ('Have fit Theil-Sen model for ' + str(num_fits_computed) + '/' +
               str(len(fit_indices)) + ' storm tracks...')

    print ('Have fit Theil-Sen model for all ' + str(len(fit_indices)) +
           ' storm tracks!')
    return storm_track_table


def _break_storm_tracks(
        storm_object_table=None, storm_track_table=None,
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
        _storm_objects_to_tracks.
    :param max_extrapolation_time_sec: Maximum extrapolation time.  No storm
        track will be extrapolated > `max_extrapolation_time_sec` before the
        time of its first storm object or after the time of its last storm
        object.
    :param max_prediction_error_metres: Maximum prediction error.  For storm
        object s to be assigned to track S, the Theil-Sen prediction for S must
        be within `max_prediction_error_metres` of the true position of s.
    :return: storm_object_table: Same as input, except that "storm_id" column
        may have changed for several objects.
    :return: storm_track_table: Same as input, except that values may have
        changed for several tracks.
    """

    # TODO(thunderhoser): Add progress messages.

    num_storm_objects = len(storm_object_table.index)

    for i in range(num_storm_objects):
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
        num_tracks_to_try = len(try_track_indices)
        x_predicted_by_track_metres = numpy.full(num_tracks_to_try, numpy.nan)
        y_predicted_by_track_metres = numpy.full(num_tracks_to_try, numpy.nan)

        for j in range(num_tracks_to_try):
            this_track_index = try_track_indices[j]
            x_predicted_by_track_metres[j], y_predicted_by_track_metres[j] = (
                _theil_sen_predict(
                    theil_sen_model_for_x=storm_track_table[
                        THEIL_SEN_MODEL_X_COLUMN].values[this_track_index],
                    theil_sen_model_for_y=storm_track_table[
                        THEIL_SEN_MODEL_Y_COLUMN].values[this_track_index],
                    query_time_unix_sec=storm_object_table[
                        tracking_io.TIME_COLUMN].values[i]))

        prediction_errors_metres = numpy.sqrt(
            (storm_object_table[CENTROID_X_COLUMN].values[i] -
             x_predicted_by_track_metres) ** 2 +
            (storm_object_table[CENTROID_Y_COLUMN].values[i] -
             y_predicted_by_track_metres) ** 2)
        if numpy.min(prediction_errors_metres) > max_prediction_error_metres:
            continue

        nearest_track_index = try_track_indices[
            numpy.argmin(prediction_errors_metres)]
        storm_object_table[tracking_io.STORM_ID_COLUMN].values[i] = (
            storm_track_table[tracking_io.STORM_ID_COLUMN].values[
                nearest_track_index])

    orig_storm_track_table = copy.deepcopy(storm_track_table)
    storm_track_table = _storm_objects_to_tracks(storm_object_table)
    storm_track_table = storm_track_table.merge(
        orig_storm_track_table, on=tracking_io.STORM_ID_COLUMN, how='left')

    track_changed_indices = _find_changed_tracks(
        storm_track_table, orig_storm_track_table)
    storm_track_table = _theil_sen_fit_for_each_track(
        storm_track_table, track_changed_indices)
    return storm_object_table, storm_track_table


def _merge_storm_tracks(
        storm_object_table=None, storm_track_table=None,
        max_join_time_sec=DEFAULT_MAX_JOIN_TIME_SEC,
        max_join_distance_metres=DEFAULT_MAX_JOIN_DISTANCE_METRES,
        max_mean_prediction_error_metres=DEFAULT_MAX_MEAN_JOIN_ERROR_METRES,
        max_velocity_diff_m_s01=None):
    """Merges pairs of similar storm tracks.

    This is the "merger" step in w2besttrack.

    :param storm_track_table: pandas DataFrame with columns documented in
        _storm_objects_to_tracks.
    :param max_join_time_sec: Maximum time between tracks (specifically, between
        end time of early track and start time of late track).
    :param max_join_distance_metres: Maximum distance between tracks
        (specifically, between last position of early track and first position
        of late track).
    :param max_velocity_diff_m_s01: Maximum difference (metres per second)
        between Theil-Sen velocities of the two tracks.  If None, this
        constraint will not be enforced.
    :return: storm_object_table: Same as input, except that "storm_id" column
        may have changed for several objects.
    :return: storm_track_table: Same as input, except that values may have
        changed for several tracks.
    """

    # TODO(thunderhoser): may want to replace `max_join_distance_metres` with
    # a speed constraint.

    num_storm_tracks = len(storm_track_table)
    remove_storm_track_flags = numpy.full(num_storm_tracks, False, dtype=bool)

    for j in range(num_storm_tracks):
        this_num_objects = len(storm_track_table[TRACK_TIMES_COLUMN].values[j])
        if this_num_objects < 2:
            continue

        for k in range(j):
            this_num_objects = len(storm_track_table[TRACK_TIMES_COLUMN].values[k])
            if this_num_objects < 2:
                continue
            if remove_storm_track_flags[k]:
                continue

            if (storm_track_table[TRACK_START_TIME_COLUMN].values[j] >=
                    storm_track_table[TRACK_START_TIME_COLUMN].values[k]):
                early_index = copy.deepcopy(k)
                late_index = copy.deepcopy(j)
            else:
                early_index = copy.deepcopy(j)
                late_index = copy.deepcopy(k)

            this_time_difference_sec = (
                storm_track_table[TRACK_START_TIME_COLUMN].values[late_index] -
                storm_track_table[TRACK_END_TIME_COLUMN].values[early_index])
            if this_time_difference_sec > max_join_time_sec:
                continue

            this_x_distance_metres = (
                storm_track_table[TRACK_X_COORDS_COLUMN].values[
                    early_index][-1] -
                storm_track_table[TRACK_X_COORDS_COLUMN].values[late_index][0])
            this_y_distance_metres = (
                storm_track_table[TRACK_Y_COORDS_COLUMN].values[
                    early_index][-1] -
                storm_track_table[TRACK_Y_COORDS_COLUMN].values[late_index][0])
            this_distance_metres = numpy.sqrt(
                this_x_distance_metres ** 2 + this_y_distance_metres ** 2)
            if this_distance_metres > max_join_distance_metres:
                continue

            if max_velocity_diff_m_s01 is None:
                this_x_velocity_diff_m_s01 = (
                    storm_track_table[
                        THEIL_SEN_MODEL_X_COLUMN].values[j].coef_[0] -
                    storm_track_table[
                        THEIL_SEN_MODEL_X_COLUMN].values[k].coef_[0])
                this_y_velocity_diff_m_s01 = (
                    storm_track_table[
                        THEIL_SEN_MODEL_Y_COLUMN].values[j].coef_[0] -
                    storm_track_table[
                        THEIL_SEN_MODEL_Y_COLUMN].values[k].coef_[0])

                this_velocity_diff_m_s01 = numpy.sqrt(
                    this_x_velocity_diff_m_s01 ** 2 +
                    this_y_velocity_diff_m_s01 ** 2)
                if this_velocity_diff_m_s01 > max_velocity_diff_m_s01:
                    continue

            num_late_objects = len(
                storm_track_table[TRACK_TIMES_COLUMN].values[late_index])
            x_late_predicted_by_early_metres = numpy.full(
                num_late_objects, numpy.nan)
            y_late_predicted_by_early_metres = numpy.full(
                num_late_objects, numpy.nan)

            for i in range(num_late_objects):
                (x_late_predicted_by_early_metres[i],
                 y_late_predicted_by_early_metres[i]) = _theil_sen_predict(
                     theil_sen_model_for_x=storm_track_table[
                         THEIL_SEN_MODEL_X_COLUMN].values[early_index],
                     theil_sen_model_for_y=storm_track_table[
                         THEIL_SEN_MODEL_Y_COLUMN].values[early_index],
                     query_time_unix_sec=storm_track_table[
                         TRACK_TIMES_COLUMN].values[late_index][i])

            these_x_errors_metres = (
                x_late_predicted_by_early_metres - storm_track_table[
                    TRACK_X_COORDS_COLUMN].values[late_index])
            these_y_errors_metres = (
                y_late_predicted_by_early_metres - storm_track_table[
                    TRACK_Y_COORDS_COLUMN].values[late_index])
            these_prediction_errors_metres = numpy.sqrt(
                these_x_errors_metres ** 2 + these_y_errors_metres ** 2)

            this_mean_error_metres = numpy.mean(these_prediction_errors_metres)
            if this_mean_error_metres > max_mean_prediction_error_metres:
                continue

            remove_storm_track_flags[k] = True
            storm_id_j = storm_track_table[
                tracking_io.STORM_ID_COLUMN].values[j]
            storm_id_k = storm_track_table[
                tracking_io.STORM_ID_COLUMN].values[k]

            for i in range(len(storm_object_table.index)):
                if storm_object_table[
                        tracking_io.STORM_ID_COLUMN].values[i] == storm_id_k:
                    storm_object_table[
                        tracking_io.STORM_ID_COLUMN].values[i] = storm_id_j

            storm_track_table_j_only = _storm_objects_to_tracks(
                storm_object_table, [storm_id_j])
            storm_track_table_j_only = _theil_sen_fit_for_each_track(
                storm_track_table_j_only)
            storm_track_table.iloc[j] = storm_track_table_j_only.iloc[0]

    return storm_object_table, storm_track_table
