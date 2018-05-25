"""Unit tests for best_tracks.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import best_tracks

TOLERANCE = 1e-6

# The following constants are used to test _theil_sen_fit.
X_COEFF_THEIL_SEN_M_S01 = 2.5
Y_COEFF_THEIL_SEN_M_S01 = -3.6

TIMES_LINEAR_UNIX_SEC = numpy.linspace(0, 9, num=10, dtype=int)
X_LINEAR_METRES = numpy.array(
    [0., 2.5, 5., 7.5, 10., 12.5, 15., 17.5, 20., 22.5])
Y_LINEAR_METRES = numpy.array(
    [0., -3.6, -7.2, -10.8, -14.4, -18., -21.6, -25.2, -28.8, -32.4])

# The following constants are used to test _theil_sen_predict.
QUERY_TIME_THEIL_SEN_UNIX_SEC = 20
X_PREDICTED_THEIL_SEN_METRES = (
    X_COEFF_THEIL_SEN_M_S01 * QUERY_TIME_THEIL_SEN_UNIX_SEC)
Y_PREDICTED_THEIL_SEN_METRES = (
    Y_COEFF_THEIL_SEN_M_S01 * QUERY_TIME_THEIL_SEN_UNIX_SEC)

# The following constants are used to test get_theil_sen_rmse.
X_ERRORS_METRES = numpy.array([-0.5, -1., -2., -1., 0., 0.5, 1., 2., 1.5, 0.])
Y_ERRORS_METRES = numpy.array([3., 1., 2., 1.5, 0.5, 0., 0.5, 1., -1., -2.])
THEIL_SEN_RMSE_METRES = numpy.sqrt(36.5 / 10)

X_NOISY_METRES = X_LINEAR_METRES + X_ERRORS_METRES
Y_NOISY_METRES = Y_LINEAR_METRES + Y_ERRORS_METRES

# The following constants are used to test
# _get_prediction_errors_for_one_object.
X_COORD_ONE_OBJECT_METRES = 5.
Y_COORD_ONE_OBJECT_METRES = 10.
TIME_ONE_OBJECT_UNIX_SEC = 1200

THESE_STORM_IDS = ['category6', 'ef12_hypercane']
THESE_START_TIMES_UNIX_SEC = numpy.array([0, 0])
THESE_END_TIMES_UNIX_SEC = numpy.array([900, 1500])

PREDICTOR_STORM_TRACK_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_STORM_IDS,
    best_tracks.TRACK_START_TIME_COLUMN: THESE_START_TIMES_UNIX_SEC,
    best_tracks.TRACK_END_TIME_COLUMN: THESE_END_TIMES_UNIX_SEC
}
PREDICTOR_STORM_TRACK_TABLE = pandas.DataFrame.from_dict(
    PREDICTOR_STORM_TRACK_DICT)

THIS_NESTED_ARRAY = PREDICTOR_STORM_TRACK_TABLE[[
    tracking_utils.STORM_ID_COLUMN,
    tracking_utils.STORM_ID_COLUMN]].values.tolist()
THIS_ARGUMENT_DICT = {
    best_tracks.TRACK_TIMES_COLUMN: THIS_NESTED_ARRAY,
    best_tracks.TRACK_X_COORDS_COLUMN: THIS_NESTED_ARRAY,
    best_tracks.TRACK_Y_COORDS_COLUMN: THIS_NESTED_ARRAY,
}
PREDICTOR_STORM_TRACK_TABLE = PREDICTOR_STORM_TRACK_TABLE.assign(
    **THIS_ARGUMENT_DICT)

PREDICTOR_STORM_TRACK_TABLE[best_tracks.TRACK_TIMES_COLUMN].values[
    0] = numpy.array([0, 300, 600, 900], dtype=int)
PREDICTOR_STORM_TRACK_TABLE[best_tracks.TRACK_TIMES_COLUMN].values[
    1] = numpy.array([0, 300, 600, 900, 1200, 1500], dtype=int)

# Theil-Sen model for this track will predict position of (20 m, 20 m) for the
# one storm object.
PREDICTOR_STORM_TRACK_TABLE[best_tracks.TRACK_X_COORDS_COLUMN].values[
    0] = numpy.array([0., 5., 10., 15.])
PREDICTOR_STORM_TRACK_TABLE[best_tracks.TRACK_Y_COORDS_COLUMN].values[
    0] = numpy.array([-20., -10., 0., 10.])

# Theil-Sen model for this track will predict position of (-5 m, 10 m) for the
# one storm object.
PREDICTOR_STORM_TRACK_TABLE[best_tracks.TRACK_X_COORDS_COLUMN].values[
    1] = numpy.array([-25., -20., -15., -10., -5., 0.])
PREDICTOR_STORM_TRACK_TABLE[best_tracks.TRACK_Y_COORDS_COLUMN].values[
    1] = numpy.array([0., 2.5, 5., 7.5, 10., 12.5])

PREDICTION_ERRORS_ONE_OBJECT_METRES = numpy.array([numpy.sqrt(325.), 10.])

# The following constants are used to test _get_join_time_for_two_tracks.
START_TIMES_OVERLAPPING_TRACKS_UNIX_SEC = numpy.array([0, 1800])
END_TIMES_OVERLAPPING_TRACKS_UNIX_SEC = numpy.array([2100, 6000])
EARLY_INDEX_OVERLAPPING_TRACKS = 0
LATE_INDEX_OVERLAPPING_TRACKS = 1

START_TIMES_NON_OVERLAPPING_TRACKS_UNIX_SEC = numpy.array([1800, 0])
END_TIMES_NON_OVERLAPPING_TRACKS_UNIX_SEC = numpy.array([6000, 1500])
EARLY_INDEX_NON_OVERLAPPING_TRACKS = 1
LATE_INDEX_NON_OVERLAPPING_TRACKS = 0
JOIN_TIME_NON_OVERLAPPING_SEC = 300

# The following constants are used to test _get_join_distance_for_two_tracks,
# _get_velocity_diff_for_two_tracks, and
# _get_mean_prediction_error_for_two_tracks.
X_COORDS_EARLY_TRACK_METRES = numpy.array([0., 5., 10.])
Y_COORDS_EARLY_TRACK_METRES = numpy.array([-5., 6., 17.])
X_COORDS_LATE_TRACK_METRES = numpy.array([20., 19., 18.])
Y_COORDS_LATE_TRACK_METRES = numpy.array([27., 25., 23.])
JOIN_DISTANCE_METRES = numpy.sqrt(200.)

TIMES_EARLY_TRACK_UNIX_SEC = numpy.array([0, 1, 2])
TIMES_LATE_TRACK_UNIX_SEC = numpy.array([7, 8, 9])
X_VELOCITY_DIFF_M_S01 = 6.
Y_VELOCITY_DIFF_M_S01 = 13.
VELOCITY_DIFFERENCE_M_S01 = numpy.sqrt(
    X_VELOCITY_DIFF_M_S01 ** 2 + Y_VELOCITY_DIFF_M_S01 ** 2)

X_LATE_PREDICTED_BY_EARLY_METRES = numpy.array([35., 40., 45.])
Y_LATE_PREDICTED_BY_EARLY_METRES = numpy.array([72., 83., 94.])
ERRORS_LATE_PREDICTED_BY_EARLY_METRES = numpy.sqrt(
    (X_LATE_PREDICTED_BY_EARLY_METRES - X_COORDS_LATE_TRACK_METRES) ** 2 +
    (Y_LATE_PREDICTED_BY_EARLY_METRES - Y_COORDS_LATE_TRACK_METRES) ** 2)

THESE_EXTRAP_TIMES_SEC = (
    TIMES_LATE_TRACK_UNIX_SEC - TIMES_EARLY_TRACK_UNIX_SEC[-1])
MEAN_ERROR_LATE_PREDICTED_BY_EARLY_M_S01 = numpy.mean(
    ERRORS_LATE_PREDICTED_BY_EARLY_METRES / THESE_EXTRAP_TIMES_SEC)

# The following constants are used to test _break_ties_one_storm_track.
X_COORDS_IN_TIE_METRES = numpy.array(
    [0., 5., 6.66, 10., -90., 15., 14.44, 20., 333.])
Y_COORDS_IN_TIE_METRES = numpy.array(
    [0., 10., -7.77, 20., 18.88, 30., 511.11, 40., 39.99])
TIMES_IN_TIE_UNIX_SEC = numpy.array([0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=int)
INDICES_TO_REMOVE_FROM_TIE = numpy.array([2, 4, 6, 8], dtype=int)

# The following constants are used to test storm_objects_to_tracks.
THESE_STORM_IDS = [
    'foo', 'bar', 'hal',
    'foo', 'bar', 'moo', best_tracks.EMPTY_STORM_ID,
    'foo', 'moo', best_tracks.EMPTY_STORM_ID]
THESE_TIMES_UNIX_SEC = numpy.array(
    [0, 0, 0,
     300, 300, 300, 300,
     600, 600, 600], dtype=int)
THESE_X_CENTROIDS_METRES = numpy.array(
    [10., 0., 20.,
     11., 1., 30., numpy.nan,
     12., 31., numpy.nan])
THESE_Y_CENTROIDS_METRES = numpy.array(
    [100., 0., 200.,
     105., 5., 300., numpy.nan,
     110., 305., numpy.nan])

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_STORM_IDS,
    tracking_utils.TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    best_tracks.CENTROID_X_COLUMN: THESE_X_CENTROIDS_METRES,
    best_tracks.CENTROID_Y_COLUMN: THESE_Y_CENTROIDS_METRES
}
MAIN_STORM_OBJECT_TABLE = pandas.DataFrame.from_dict(THIS_DICT)

THESE_STORM_IDS = ['bar', 'foo', 'hal', 'moo']
THESE_START_TIMES_UNIX_SEC = numpy.array([0, 0, 0, 300], dtype=int)
THESE_END_TIMES_UNIX_SEC = numpy.array([300, 600, 0, 600], dtype=int)

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_STORM_IDS,
    best_tracks.TRACK_START_TIME_COLUMN: THESE_START_TIMES_UNIX_SEC,
    best_tracks.TRACK_END_TIME_COLUMN: THESE_END_TIMES_UNIX_SEC
}
MAIN_STORM_TRACK_TABLE = pandas.DataFrame.from_dict(THIS_DICT)

THIS_NESTED_ARRAY = MAIN_STORM_TRACK_TABLE[[
    tracking_utils.STORM_ID_COLUMN,
    tracking_utils.STORM_ID_COLUMN]].values.tolist()
THIS_ARGUMENT_DICT = {
    best_tracks.TRACK_TIMES_COLUMN: THIS_NESTED_ARRAY,
    best_tracks.TRACK_X_COORDS_COLUMN: THIS_NESTED_ARRAY,
    best_tracks.TRACK_Y_COORDS_COLUMN: THIS_NESTED_ARRAY,
    best_tracks.OBJECT_INDICES_COLUMN_FOR_TRACK: THIS_NESTED_ARRAY
}
MAIN_STORM_TRACK_TABLE = MAIN_STORM_TRACK_TABLE.assign(**THIS_ARGUMENT_DICT)

MAIN_STORM_TRACK_TABLE[best_tracks.OBJECT_INDICES_COLUMN_FOR_TRACK].values[
    0] = numpy.array([1, 4], dtype=int)
MAIN_STORM_TRACK_TABLE[best_tracks.OBJECT_INDICES_COLUMN_FOR_TRACK].values[
    1] = numpy.array([0, 3, 7], dtype=int)
MAIN_STORM_TRACK_TABLE[best_tracks.OBJECT_INDICES_COLUMN_FOR_TRACK].values[
    2] = numpy.array([2], dtype=int)
MAIN_STORM_TRACK_TABLE[best_tracks.OBJECT_INDICES_COLUMN_FOR_TRACK].values[
    3] = numpy.array([5, 8], dtype=int)

MAIN_STORM_TRACK_TABLE[best_tracks.TRACK_TIMES_COLUMN].values[
    0] = numpy.array([0, 300], dtype=int)
MAIN_STORM_TRACK_TABLE[best_tracks.TRACK_TIMES_COLUMN].values[
    1] = numpy.array([0, 300, 600], dtype=int)
MAIN_STORM_TRACK_TABLE[best_tracks.TRACK_TIMES_COLUMN].values[
    2] = numpy.array([0], dtype=int)
MAIN_STORM_TRACK_TABLE[best_tracks.TRACK_TIMES_COLUMN].values[
    3] = numpy.array([300, 600], dtype=int)

MAIN_STORM_TRACK_TABLE[best_tracks.TRACK_X_COORDS_COLUMN].values[
    0] = numpy.array([0., 1.])
MAIN_STORM_TRACK_TABLE[best_tracks.TRACK_Y_COORDS_COLUMN].values[
    0] = numpy.array([0., 5.])
MAIN_STORM_TRACK_TABLE[best_tracks.TRACK_X_COORDS_COLUMN].values[
    1] = numpy.array([10., 11., 12.])
MAIN_STORM_TRACK_TABLE[best_tracks.TRACK_Y_COORDS_COLUMN].values[
    1] = numpy.array([100., 105., 110.])
MAIN_STORM_TRACK_TABLE[best_tracks.TRACK_X_COORDS_COLUMN].values[
    2] = numpy.array([20.])
MAIN_STORM_TRACK_TABLE[best_tracks.TRACK_Y_COORDS_COLUMN].values[
    2] = numpy.array([200.])
MAIN_STORM_TRACK_TABLE[best_tracks.TRACK_X_COORDS_COLUMN].values[
    3] = numpy.array([30., 31.])
MAIN_STORM_TRACK_TABLE[best_tracks.TRACK_Y_COORDS_COLUMN].values[
    3] = numpy.array([300., 305.])

STORM_TRACK_TABLE_FOO_ONLY = MAIN_STORM_TRACK_TABLE.loc[
    MAIN_STORM_TRACK_TABLE[tracking_utils.STORM_ID_COLUMN] == 'foo']

ARRAY_COLUMNS_IN_STORM_TRACK_TABLE = [
    best_tracks.TRACK_TIMES_COLUMN, best_tracks.TRACK_X_COORDS_COLUMN,
    best_tracks.TRACK_Y_COORDS_COLUMN,
    best_tracks.OBJECT_INDICES_COLUMN_FOR_TRACK]
STRING_COLUMNS_IN_STORM_TRACK_TABLE = [tracking_utils.STORM_ID_COLUMN]

# The following constants are used to test _find_changed_tracks.
STORM_TRACK_TABLE_CHANGED = copy.deepcopy(MAIN_STORM_TRACK_TABLE)
STORM_TRACK_TABLE_CHANGED[best_tracks.TRACK_X_COORDS_COLUMN].values[
    0] = numpy.array([123., 1.])
STORM_TRACK_TABLE_CHANGED[best_tracks.TRACK_TIMES_COLUMN].values[
    1] = numpy.array([0, 303, 600], dtype=int)
STORM_TRACK_TABLE_CHANGED[best_tracks.TRACK_Y_COORDS_COLUMN].values[
    3] = numpy.array([300., 321.])

TRACK_CHANGED_INDICES = numpy.array([0, 1, 3], dtype=int)

# The following constants are used to test remove_short_tracks.
ROWS_WITH_TRACK_LENGTH_LESS_THAN1 = numpy.array([6, 9], dtype=int)
ROWS_WITH_TRACK_LENGTH_LESS_THAN2 = numpy.array([2, 6, 9], dtype=int)
ROWS_WITH_TRACK_LENGTH_LESS_THAN3 = numpy.array(
    [1, 2, 4, 5, 6, 8, 9], dtype=int)

STORM_OBJECT_TABLE_TRACK_LENGTH_GEQ1 = MAIN_STORM_OBJECT_TABLE.drop(
    MAIN_STORM_OBJECT_TABLE.index[ROWS_WITH_TRACK_LENGTH_LESS_THAN1], axis=0,
    inplace=False)
STORM_OBJECT_TABLE_TRACK_LENGTH_GEQ2 = MAIN_STORM_OBJECT_TABLE.drop(
    MAIN_STORM_OBJECT_TABLE.index[ROWS_WITH_TRACK_LENGTH_LESS_THAN2], axis=0,
    inplace=False)
STORM_OBJECT_TABLE_TRACK_LENGTH_GEQ3 = MAIN_STORM_OBJECT_TABLE.drop(
    MAIN_STORM_OBJECT_TABLE.index[ROWS_WITH_TRACK_LENGTH_LESS_THAN3], axis=0,
    inplace=False)

# The following constants are used to test recompute_attributes.
BEST_TRACK_START_TIME_UNIX_SEC = 0
BEST_TRACK_END_TIME_UNIX_SEC = 600

TRACKING_START_TIME_BY_OBJECT_UNIX_SEC = numpy.full(10, 0, dtype=int)
TRACKING_END_TIME_BY_OBJECT_UNIX_SEC = numpy.full(10, 600, dtype=int)
TRACK_AGE_BY_OBJECT_SEC = numpy.array(
    [-1, -1, -1, -1, -1, 0, 0, -1, 300, 300], dtype=int)
CELL_START_TIME_BY_OBJECT_UNIX_SEC = numpy.array(
    [0, 0, 0, 0, 0, 300, 300, 0, 300, 300], dtype=int)
CELL_END_TIME_BY_OBJECT_UNIX_SEC = numpy.array(
    [600, 300, 0, 600, 300, 600, 600, 600, 600, 600], dtype=int)


class BestTracksTests(unittest.TestCase):
    """Each method is a unit test for best_tracks.py."""

    def test_theil_sen_fit(self):
        """Ensures correct output from _theil_sen_fit."""

        this_ts_model_for_x, this_ts_model_for_y = best_tracks._theil_sen_fit(
            unix_times_sec=TIMES_LINEAR_UNIX_SEC,
            x_coords_metres=X_LINEAR_METRES, y_coords_metres=Y_LINEAR_METRES)

        this_x_coefficient_m_s01 = this_ts_model_for_x.coef_[0]
        this_y_coefficient_m_s01 = this_ts_model_for_y.coef_[0]

        self.assertTrue(
            numpy.absolute(this_x_coefficient_m_s01 - X_COEFF_THEIL_SEN_M_S01)
            <= TOLERANCE)
        self.assertTrue(
            numpy.absolute(this_y_coefficient_m_s01 - Y_COEFF_THEIL_SEN_M_S01)
            <= TOLERANCE)

    def test_theil_sen_predict(self):
        """Ensures correct output from _theil_sen_predict.

        This is an integration test, not a unit test, because it also requires
        _theil_sen_fit.
        """

        this_ts_model_for_x, this_ts_model_for_y = best_tracks._theil_sen_fit(
            unix_times_sec=TIMES_LINEAR_UNIX_SEC,
            x_coords_metres=X_LINEAR_METRES, y_coords_metres=Y_LINEAR_METRES)

        this_x_predicted_metres, this_y_predicted_metres = (
            best_tracks._theil_sen_predict(
                theil_sen_model_for_x=this_ts_model_for_x,
                theil_sen_model_for_y=this_ts_model_for_y,
                query_time_unix_sec=QUERY_TIME_THEIL_SEN_UNIX_SEC))

        self.assertTrue(
            numpy.absolute(this_x_predicted_metres -
                           X_PREDICTED_THEIL_SEN_METRES) <= TOLERANCE)
        self.assertTrue(
            numpy.absolute(this_y_predicted_metres -
                           Y_PREDICTED_THEIL_SEN_METRES) <= TOLERANCE)

    def test_get_theil_sen_rmse(self):
        """Ensures correct output from _get_theil_sen_rmse.

        This is an integration test, not a unit test, because it also requires
        _theil_sen_fit.
        """

        this_ts_model_for_x, this_ts_model_for_y = best_tracks._theil_sen_fit(
            unix_times_sec=TIMES_LINEAR_UNIX_SEC,
            x_coords_metres=X_LINEAR_METRES, y_coords_metres=Y_LINEAR_METRES)

        this_rmse_metres = best_tracks._get_theil_sen_rmse(
            theil_sen_model_for_x=this_ts_model_for_x,
            theil_sen_model_for_y=this_ts_model_for_y,
            track_times_unix_sec=TIMES_LINEAR_UNIX_SEC,
            track_x_metres=X_NOISY_METRES, track_y_metres=Y_NOISY_METRES)

        self.assertTrue(numpy.isclose(this_rmse_metres, THEIL_SEN_RMSE_METRES))

    def test_get_prediction_errors_for_one_object(self):
        """Ensures correct output from _get_prediction_errors_for_one_object.

        This is an integration test, not a unit test, because it depends on
        _theil_sen_fit_for_each_track.
        """

        this_storm_track_table = best_tracks.theil_sen_fit_for_each_track(
            PREDICTOR_STORM_TRACK_TABLE)

        these_prediction_errors_metres = (
            best_tracks._get_prediction_errors_for_one_object(
                x_coord_metres=X_COORD_ONE_OBJECT_METRES,
                y_coord_metres=Y_COORD_ONE_OBJECT_METRES,
                unix_time_sec=TIME_ONE_OBJECT_UNIX_SEC,
                storm_track_table=this_storm_track_table))

        self.assertTrue(numpy.allclose(
            these_prediction_errors_metres, PREDICTION_ERRORS_ONE_OBJECT_METRES,
            atol=TOLERANCE))

    def test_get_join_time_for_two_tracks_overlapping(self):
        """Ensures correct output from _get_join_time_for_two_tracks.

        In this case the two tracks overlap in time.
        """

        this_join_time_sec, this_early_index, this_late_index = (
            best_tracks._get_join_time_for_two_tracks(
                START_TIMES_OVERLAPPING_TRACKS_UNIX_SEC,
                END_TIMES_OVERLAPPING_TRACKS_UNIX_SEC))

        self.assertTrue(numpy.isnan(this_join_time_sec))
        self.assertTrue(this_early_index == EARLY_INDEX_OVERLAPPING_TRACKS)
        self.assertTrue(this_late_index == LATE_INDEX_OVERLAPPING_TRACKS)

    def test_get_join_time_for_two_tracks_non_overlapping(self):
        """Ensures correct output from _get_join_time_for_two_tracks.

        In this case the two tracks do not overlap in time.
        """

        this_join_time_sec, this_early_index, this_late_index = (
            best_tracks._get_join_time_for_two_tracks(
                START_TIMES_NON_OVERLAPPING_TRACKS_UNIX_SEC,
                END_TIMES_NON_OVERLAPPING_TRACKS_UNIX_SEC))

        self.assertTrue(this_join_time_sec == JOIN_TIME_NON_OVERLAPPING_SEC)
        self.assertTrue(this_early_index == EARLY_INDEX_NON_OVERLAPPING_TRACKS)
        self.assertTrue(this_late_index == LATE_INDEX_NON_OVERLAPPING_TRACKS)

    def test_get_join_distance_for_two_tracks(self):
        """Ensures correct output from _get_join_distance_for_two_tracks."""

        this_join_distance_metres = (
            best_tracks._get_join_distance_for_two_tracks(
                x_coords_early_metres=X_COORDS_EARLY_TRACK_METRES,
                y_coords_early_metres=Y_COORDS_EARLY_TRACK_METRES,
                x_coords_late_metres=X_COORDS_LATE_TRACK_METRES,
                y_coords_late_metres=Y_COORDS_LATE_TRACK_METRES))
        self.assertTrue(numpy.isclose(
            this_join_distance_metres, JOIN_DISTANCE_METRES))

    def test_get_velocity_diff_for_two_tracks(self):
        """Ensures correct output from _get_velocity_diff_for_two_tracks.

        This is an integration test, not a unit test, because it depends on
        _theil_sen_fit.
        """

        theil_sen_model_for_x_early, theil_sen_model_for_y_early = (
            best_tracks._theil_sen_fit(
                unix_times_sec=TIMES_EARLY_TRACK_UNIX_SEC,
                x_coords_metres=X_COORDS_EARLY_TRACK_METRES,
                y_coords_metres=Y_COORDS_EARLY_TRACK_METRES))

        theil_sen_model_for_x_late, theil_sen_model_for_y_late = (
            best_tracks._theil_sen_fit(
                unix_times_sec=TIMES_LATE_TRACK_UNIX_SEC,
                x_coords_metres=X_COORDS_LATE_TRACK_METRES,
                y_coords_metres=Y_COORDS_LATE_TRACK_METRES))

        this_velocity_diff_m_s01 = (
            best_tracks._get_velocity_diff_for_two_tracks(
                [theil_sen_model_for_x_early, theil_sen_model_for_x_late],
                [theil_sen_model_for_y_early, theil_sen_model_for_y_late]))
        self.assertTrue(numpy.isclose(
            this_velocity_diff_m_s01, VELOCITY_DIFFERENCE_M_S01,
            atol=TOLERANCE))

    def test_get_mean_prediction_error_for_two_tracks(self):
        """Ensures correctness of _get_mean_prediction_error_for_two_tracks.

        This is an integration test, not a unit test, because it depends on
        _theil_sen_fit.
        """

        theil_sen_model_for_x_early, theil_sen_model_for_y_early = (
            best_tracks._theil_sen_fit(
                unix_times_sec=TIMES_EARLY_TRACK_UNIX_SEC,
                x_coords_metres=X_COORDS_EARLY_TRACK_METRES,
                y_coords_metres=Y_COORDS_EARLY_TRACK_METRES))

        this_mean_error_m_s01 = (
            best_tracks._get_mean_prediction_error_for_two_tracks(
                x_coords_late_metres=X_COORDS_LATE_TRACK_METRES,
                y_coords_late_metres=Y_COORDS_LATE_TRACK_METRES,
                late_times_unix_sec=TIMES_LATE_TRACK_UNIX_SEC,
                theil_sen_model_for_x_early=theil_sen_model_for_x_early,
                theil_sen_model_for_y_early=theil_sen_model_for_y_early,
                last_early_time_unix_sec=numpy.max(TIMES_EARLY_TRACK_UNIX_SEC)))

        self.assertTrue(numpy.isclose(
            this_mean_error_m_s01, MEAN_ERROR_LATE_PREDICTED_BY_EARLY_M_S01,
            atol=TOLERANCE))

    def test_break_ties_one_storm_track(self):
        """Ensures correct output from _break_ties_one_storm_track."""

        this_theil_sen_model_for_x, this_theil_sen_model_for_y = (
            best_tracks._theil_sen_fit(
                unix_times_sec=TIMES_IN_TIE_UNIX_SEC,
                x_coords_metres=X_COORDS_IN_TIE_METRES,
                y_coords_metres=Y_COORDS_IN_TIE_METRES))

        these_indices_to_remove = best_tracks._break_ties_one_storm_track(
            object_x_coords_metres=X_COORDS_IN_TIE_METRES,
            object_y_coords_metres=Y_COORDS_IN_TIE_METRES,
            object_times_unix_sec=TIMES_IN_TIE_UNIX_SEC,
            theil_sen_model_for_x=this_theil_sen_model_for_x,
            theil_sen_model_for_y=this_theil_sen_model_for_y)

        self.assertTrue(numpy.array_equal(
            these_indices_to_remove, INDICES_TO_REMOVE_FROM_TIE))

    def test_find_changed_tracks(self):
        """Ensures correct output from _find_changed_tracks."""

        these_track_changed_indices = best_tracks._find_changed_tracks(
            STORM_TRACK_TABLE_CHANGED, MAIN_STORM_TRACK_TABLE)
        self.assertTrue(numpy.array_equal(
            these_track_changed_indices, TRACK_CHANGED_INDICES))

    def test_storm_objects_to_tracks_all_storms(self):
        """Ensures correct output from storm_objects_to_tracks.

        In this case, tracking info is added for all storms.
        """

        this_storm_track_table = best_tracks.storm_objects_to_tracks(
            MAIN_STORM_OBJECT_TABLE, storm_ids_to_use=None)
        this_storm_track_table.sort_values(
            tracking_utils.STORM_ID_COLUMN, axis=0, ascending=True,
            inplace=True)

        self.assertTrue(set(list(this_storm_track_table)) ==
                        set(list(MAIN_STORM_TRACK_TABLE)))
        num_rows = len(MAIN_STORM_TRACK_TABLE.index)

        for this_column in list(MAIN_STORM_TRACK_TABLE):
            if this_column in STRING_COLUMNS_IN_STORM_TRACK_TABLE:
                self.assertTrue(numpy.array_equal(
                    this_storm_track_table[this_column].values,
                    MAIN_STORM_TRACK_TABLE[this_column].values))

            elif this_column in ARRAY_COLUMNS_IN_STORM_TRACK_TABLE:
                for i in range(num_rows):
                    self.assertTrue(numpy.allclose(
                        this_storm_track_table[this_column].values[i],
                        MAIN_STORM_TRACK_TABLE[this_column].values[i],
                        atol=TOLERANCE))

            else:
                self.assertTrue(numpy.allclose(
                    this_storm_track_table[this_column].values,
                    MAIN_STORM_TRACK_TABLE[this_column].values, atol=TOLERANCE))

    def test_storm_objects_to_tracks_foo_only(self):
        """Ensures correct output from storm_objects_to_tracks.

        In this case, tracking info is added only for storm "foo".
        """

        this_storm_track_table = best_tracks.storm_objects_to_tracks(
            MAIN_STORM_OBJECT_TABLE, storm_ids_to_use=['foo'])

        self.assertTrue(set(list(this_storm_track_table)) ==
                        set(list(STORM_TRACK_TABLE_FOO_ONLY)))
        num_rows = len(STORM_TRACK_TABLE_FOO_ONLY.index)

        for this_column in list(STORM_TRACK_TABLE_FOO_ONLY):
            if this_column in STRING_COLUMNS_IN_STORM_TRACK_TABLE:
                self.assertTrue(numpy.array_equal(
                    this_storm_track_table[this_column].values,
                    STORM_TRACK_TABLE_FOO_ONLY[this_column].values))

            elif this_column in ARRAY_COLUMNS_IN_STORM_TRACK_TABLE:
                for i in range(num_rows):
                    self.assertTrue(numpy.allclose(
                        this_storm_track_table[this_column].values[i],
                        STORM_TRACK_TABLE_FOO_ONLY[this_column].values[i],
                        atol=TOLERANCE))

            else:
                self.assertTrue(numpy.allclose(
                    this_storm_track_table[this_column].values,
                    STORM_TRACK_TABLE_FOO_ONLY[this_column].values,
                    atol=TOLERANCE))

    def test_remove_short_tracks_min_length1(self):
        """Ensures correct output from remove_short_tracks.

        In this case, minimum track length is one storm object.
        """

        this_storm_object_table = best_tracks.remove_short_tracks(
            MAIN_STORM_OBJECT_TABLE, min_objects_in_track=1)

        self.assertTrue(numpy.array_equal(
            this_storm_object_table[tracking_utils.STORM_ID_COLUMN].values,
            STORM_OBJECT_TABLE_TRACK_LENGTH_GEQ1[
                tracking_utils.STORM_ID_COLUMN].values))
        self.assertTrue(numpy.array_equal(
            this_storm_object_table[tracking_utils.TIME_COLUMN].values,
            STORM_OBJECT_TABLE_TRACK_LENGTH_GEQ1[
                tracking_utils.TIME_COLUMN].values))

    def test_remove_short_tracks_min_length2(self):
        """Ensures correct output from remove_short_tracks.

        In this case, minimum track length is 2 storm objects.
        """

        this_storm_object_table = best_tracks.remove_short_tracks(
            MAIN_STORM_OBJECT_TABLE, min_objects_in_track=2)

        self.assertTrue(numpy.array_equal(
            this_storm_object_table[tracking_utils.STORM_ID_COLUMN].values,
            STORM_OBJECT_TABLE_TRACK_LENGTH_GEQ2[
                tracking_utils.STORM_ID_COLUMN].values))
        self.assertTrue(numpy.array_equal(
            this_storm_object_table[tracking_utils.TIME_COLUMN].values,
            STORM_OBJECT_TABLE_TRACK_LENGTH_GEQ2[
                tracking_utils.TIME_COLUMN].values))

    def test_remove_short_tracks_min_length3(self):
        """Ensures correct output from remove_short_tracks.

        In this case, minimum track length is 3 storm objects.
        """

        this_storm_object_table = best_tracks.remove_short_tracks(
            MAIN_STORM_OBJECT_TABLE, min_objects_in_track=3)

        self.assertTrue(numpy.array_equal(
            this_storm_object_table[tracking_utils.STORM_ID_COLUMN].values,
            STORM_OBJECT_TABLE_TRACK_LENGTH_GEQ3[
                tracking_utils.STORM_ID_COLUMN].values))
        self.assertTrue(numpy.array_equal(
            this_storm_object_table[tracking_utils.TIME_COLUMN].values,
            STORM_OBJECT_TABLE_TRACK_LENGTH_GEQ3[
                tracking_utils.TIME_COLUMN].values))

    def test_recompute_attributes(self):
        """Ensures correct output from recompute_attributes."""

        this_storm_object_table = best_tracks.recompute_attributes(
            MAIN_STORM_OBJECT_TABLE,
            best_track_start_time_unix_sec=BEST_TRACK_START_TIME_UNIX_SEC,
            best_track_end_time_unix_sec=BEST_TRACK_END_TIME_UNIX_SEC)

        self.assertTrue(numpy.array_equal(
            this_storm_object_table[
                tracking_utils.TRACKING_START_TIME_COLUMN].values,
            TRACKING_START_TIME_BY_OBJECT_UNIX_SEC))
        self.assertTrue(numpy.array_equal(
            this_storm_object_table[
                tracking_utils.TRACKING_END_TIME_COLUMN].values,
            TRACKING_END_TIME_BY_OBJECT_UNIX_SEC))
        self.assertTrue(numpy.array_equal(
            this_storm_object_table[tracking_utils.AGE_COLUMN].values,
            TRACK_AGE_BY_OBJECT_SEC))
        self.assertTrue(numpy.array_equal(
            this_storm_object_table[
                tracking_utils.CELL_START_TIME_COLUMN].values,
            CELL_START_TIME_BY_OBJECT_UNIX_SEC))
        self.assertTrue(numpy.array_equal(
            this_storm_object_table[tracking_utils.CELL_END_TIME_COLUMN].values,
            CELL_END_TIME_BY_OBJECT_UNIX_SEC))


if __name__ == '__main__':
    unittest.main()
