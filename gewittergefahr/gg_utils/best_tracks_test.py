"""Unit tests for best_tracks.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import best_tracks

TOLERANCE = 1e-6

# The following constants are used to test _theil_sen_fit_one_track.
X_INTERCEPT_ONE_TRACK_METRES = 0.
X_VELOCITY_ONE_TRACK_M_S01 = 2.5
Y_INTERCEPT_ONE_TRACK_METRES = 0.
Y_VELOCITY_ONE_TRACK_M_S01 = -3.6

VALID_TIMES_ONE_TRACK_UNIX_SEC = numpy.linspace(0, 9, num=10, dtype=int)
X_COORDS_ONE_TRACK_METRES = numpy.array(
    [0., 2.5, 5., 7.5, 10., 12.5, 15., 17.5, 20., 22.5])
Y_COORDS_ONE_TRACK_METRES = numpy.array(
    [0., -3.6, -7.2, -10.8, -14.4, -18., -21.6, -25.2, -28.8, -32.4])

# The following constants are used to test _theil_sen_predict and
# _theil_sen_predict_many_times.
QUERY_TIMES_ONE_TRACK_UNIX_SEC = numpy.array([-10, 0, 10, 20], dtype=int)
QUERY_X_COORDS_ONE_TRACK_METRES = numpy.array([-25, 0, 25, 50], dtype=float)
QUERY_Y_COORDS_ONE_TRACK_METRES = numpy.array([36, 0, -36, -72], dtype=float)

# The following constants are used to test _theil_sen_predict_many_models.
X_INTERCEPTS_MANY_MODELS_METRES = numpy.array([-5, 0, 5, 10], dtype=float)
X_VELOCITIES_MANY_MODELS_M_S01 = numpy.array([-5, 0, 5, 10], dtype=float)
Y_INTERCEPTS_MANY_MODELS_METRES = numpy.array([1, 2, 3, 4], dtype=float)
Y_VELOCITIES_MANY_MODELS_M_S01 = numpy.array([-1, -2, -3, -4], dtype=float)
QUERY_TIME_MANY_MODELS_UNIX_SEC = 7

QUERY_X_COORDS_MANY_MODELS_METRES = numpy.array([-40, 0, 40, 80], dtype=float)
QUERY_Y_COORDS_MANY_MODELS_METRES = numpy.array(
    [-6, -12, -18, -24], dtype=float)

# The following constants are used to test _get_theil_sen_error_one_track and
# _get_theil_sen_errors_one_object.
THESE_X_ERRORS_METRES = numpy.array(
    [-0.5, -1., -2., -1., 0., 0.5, 1., 2., 1.5, 0.])
THESE_Y_ERRORS_METRES = numpy.array(
    [3., 1., 2., 1.5, 0.5, 0., 0.5, 1., -1., -2.])

ONE_TRACK_RMSE_METRES = numpy.sqrt(36.5 / 10)
ONE_TRACK_FIRST_OBJECT_ERROR_METRES = numpy.sqrt(9.25)

THIS_DICT = {
    best_tracks.THEIL_SEN_X_INTERCEPT_COLUMN: X_INTERCEPT_ONE_TRACK_METRES,
    best_tracks.THEIL_SEN_X_VELOCITY_COLUMN: X_VELOCITY_ONE_TRACK_M_S01,
    best_tracks.THEIL_SEN_Y_INTERCEPT_COLUMN: Y_INTERCEPT_ONE_TRACK_METRES,
    best_tracks.THEIL_SEN_Y_VELOCITY_COLUMN: Y_VELOCITY_ONE_TRACK_M_S01,
    best_tracks.TRACK_TIMES_COLUMN: [VALID_TIMES_ONE_TRACK_UNIX_SEC],
    best_tracks.TRACK_X_COORDS_COLUMN:
        [X_COORDS_ONE_TRACK_METRES + THESE_X_ERRORS_METRES],
    best_tracks.TRACK_Y_COORDS_COLUMN:
        [Y_COORDS_ONE_TRACK_METRES + THESE_Y_ERRORS_METRES]
}
ONE_TRACK_TABLE_NOISY = pandas.DataFrame.from_dict(THIS_DICT)

# The following constants are used to test _get_join_time_for_two_tracks.
START_TIMES_OVERLAPPING_TRACKS_UNIX_SEC = numpy.array([0, 1800], dtype=int)
END_TIMES_OVERLAPPING_TRACKS_UNIX_SEC = numpy.array([2100, 6000], dtype=int)
EARLY_INDEX_OVERLAPPING_TRACKS = 0
LATE_INDEX_OVERLAPPING_TRACKS = 1

START_TIMES_NON_OVERLAPPING_TRACKS_UNIX_SEC = numpy.array([1800, 0], dtype=int)
END_TIMES_NON_OVERLAPPING_TRACKS_UNIX_SEC = numpy.array([6000, 1500], dtype=int)
EARLY_INDEX_NON_OVERLAPPING_TRACKS = 1
LATE_INDEX_NON_OVERLAPPING_TRACKS = 0
JOIN_TIME_NON_OVERLAPPING_SEC = 300

# The following constants are used to test _get_join_distance_for_two_tracks.
X_COORDS_EARLY_TRACK_METRES = numpy.array([0, 5, 10], dtype=float)
Y_COORDS_EARLY_TRACK_METRES = numpy.array([-5, 6, 17], dtype=float)
X_COORDS_LATE_TRACK_METRES = numpy.array([20, 19, 18], dtype=float)
Y_COORDS_LATE_TRACK_METRES = numpy.array([27, 25, 23], dtype=float)
JOIN_DISTANCE_TWO_TRACKS_METRES = numpy.sqrt(200.)

# The following constants are used to test _get_velocity_difference_two_tracks.
VALID_TIMES_EARLY_TRACK_UNIX_SEC = numpy.array([0, 1, 2], dtype=int)
VALID_TIMES_LATE_TRACK_UNIX_SEC = numpy.array([7, 8, 9], dtype=int)
VELOCITY_DIFF_TWO_TRACKS_M_S01 = numpy.sqrt(6**2 + 13**2)

# The following constants are used to test _get_theil_sen_error_two_tracks.
THESE_X_PREDICTED_METRES = numpy.array([35, 40, 45], dtype=float)
THESE_Y_PREDICTED_METRES = numpy.array([72, 83, 94], dtype=float)
THESE_ERRORS_METRES = numpy.sqrt(
    (THESE_X_PREDICTED_METRES - X_COORDS_LATE_TRACK_METRES) ** 2 +
    (THESE_Y_PREDICTED_METRES - Y_COORDS_LATE_TRACK_METRES) ** 2)

THESE_EXTRAP_TIMES_SECONDS = (
    VALID_TIMES_LATE_TRACK_UNIX_SEC - VALID_TIMES_EARLY_TRACK_UNIX_SEC[-1])
MEAN_ERROR_TWO_TRACKS_M_S01 = numpy.mean(
    THESE_ERRORS_METRES / THESE_EXTRAP_TIMES_SECONDS)

THIS_DICT = {
    best_tracks.TRACK_X_COORDS_COLUMN:
        [X_COORDS_LATE_TRACK_METRES, X_COORDS_EARLY_TRACK_METRES],
    best_tracks.TRACK_Y_COORDS_COLUMN:
        [Y_COORDS_LATE_TRACK_METRES, Y_COORDS_EARLY_TRACK_METRES],
    best_tracks.TRACK_TIMES_COLUMN:
        [VALID_TIMES_LATE_TRACK_UNIX_SEC, VALID_TIMES_EARLY_TRACK_UNIX_SEC]
}
TWO_STORM_TRACKS_TABLE = pandas.DataFrame.from_dict(THIS_DICT)

# The following constants are used to test _break_ties_one_track.
THESE_X_COORDS_METRES = numpy.array(
    [0., 5., 6.66, 10., -90., 15., 14.44, 20., 333.])
THESE_Y_COORDS_METRES = numpy.array(
    [0., 10., -7.77, 20., 18.88, 30., 511.11, 40., 39.99])
THESE_TIMES_UNIX_SEC = numpy.array([0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=int)
INDICES_TO_REMOVE_FOR_TIEBREAKER = numpy.array([2, 4, 6, 8], dtype=int)

THIS_DICT = {
    best_tracks.TRACK_X_COORDS_COLUMN: [THESE_X_COORDS_METRES],
    best_tracks.TRACK_Y_COORDS_COLUMN: [THESE_Y_COORDS_METRES],
    best_tracks.TRACK_TIMES_COLUMN: [THESE_TIMES_UNIX_SEC]
}
STORM_TRACK_TABLE_FOR_TIEBREAKER = pandas.DataFrame.from_dict(THIS_DICT)

# The following constants are used to test storm_objects_to_tracks.
THESE_STORM_IDS = [
    'foo', 'bar', 'hal', 'foo', 'bar', 'moo', best_tracks.EMPTY_STORM_ID, 'foo',
    'moo', best_tracks.EMPTY_STORM_ID]
THESE_TIMES_UNIX_SEC = numpy.array(
    [0, 0, 0, 300, 300, 300, 300, 600, 600, 600], dtype=int)
THESE_X_COORDS_METRES = numpy.array(
    [10, 0, 20, 11, 1, 30, numpy.nan, 12, 31, numpy.nan])
THESE_Y_COORDS_METRES = numpy.array(
    [100, 0, 200, 105, 5, 300, numpy.nan, 110, 305, numpy.nan])

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_STORM_IDS,
    tracking_utils.TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    best_tracks.CENTROID_X_COLUMN: THESE_X_COORDS_METRES,
    best_tracks.CENTROID_Y_COLUMN: THESE_Y_COORDS_METRES
}
MAIN_STORM_OBJECT_TABLE = pandas.DataFrame.from_dict(THIS_DICT)

THESE_STORM_IDS = ['bar', 'foo', 'hal', 'moo']
THESE_START_TIMES_UNIX_SEC = numpy.array([0, 0, 0, 300], dtype=int)
THESE_END_TIMES_UNIX_SEC = numpy.array([300, 600, 0, 600], dtype=int)

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_STORM_IDS,
    best_tracks.TRACK_START_TIME_COLUMN: THESE_START_TIMES_UNIX_SEC,
    best_tracks.TRACK_END_TIME_COLUMN: THESE_END_TIMES_UNIX_SEC,
    best_tracks.TRACK_TIMES_COLUMN: [[0, 300], [0, 300, 600], [0], [300, 600]],
    best_tracks.TRACK_X_COORDS_COLUMN: [[0, 1], [10, 11, 12], [20], [30, 31]],
    best_tracks.TRACK_Y_COORDS_COLUMN:
        [[0, 5], [100, 105, 110], [200], [300, 305]],
    best_tracks.OBJECT_INDICES_COLUMN_FOR_TRACK:
        [[1, 4], [0, 3, 7], [2], [5, 8]]
}
MAIN_STORM_TRACK_TABLE = pandas.DataFrame.from_dict(THIS_DICT)

STORM_TRACK_TABLE_FOO_ONLY = MAIN_STORM_TRACK_TABLE.loc[
    MAIN_STORM_TRACK_TABLE[tracking_utils.STORM_ID_COLUMN] == 'foo']

ARRAY_COLUMNS_IN_TRACK_TABLE = [
    best_tracks.TRACK_TIMES_COLUMN, best_tracks.TRACK_X_COORDS_COLUMN,
    best_tracks.TRACK_Y_COORDS_COLUMN,
    best_tracks.OBJECT_INDICES_COLUMN_FOR_TRACK]
STRING_COLUMNS_IN_TRACK_TABLE = [tracking_utils.STORM_ID_COLUMN]

# The following constants are used to test _find_changed_tracks.
MAIN_STORM_TRACK_TABLE_TWEAKED = copy.deepcopy(MAIN_STORM_TRACK_TABLE)
MAIN_STORM_TRACK_TABLE_TWEAKED[
    best_tracks.TRACK_X_COORDS_COLUMN
].values[0] = numpy.array([123, 1], dtype=float)

MAIN_STORM_TRACK_TABLE_TWEAKED[
    best_tracks.TRACK_TIMES_COLUMN
].values[1] = numpy.array([0, 303, 600], dtype=int)

MAIN_STORM_TRACK_TABLE_TWEAKED[
    best_tracks.TRACK_Y_COORDS_COLUMN
].values[3] = numpy.array([300, 305], dtype=float)

TRACK_CHANGED_INDICES = numpy.array([0, 1], dtype=int)

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

# The following constants are used to test get_storm_ages.
THESE_STORM_IDS = [
    'Ricky', 'Julian', 'Bubbles', 'Trinity', 'Julian', 'Ricky',
    best_tracks.EMPTY_STORM_ID, best_tracks.EMPTY_STORM_ID, 'Ricky'
]
THESE_TIMES_UNIX_SEC = numpy.array(
    [0, 0, 0, 300, 300, 300, 300, 600, 600], dtype=int)

BEST_TRACK_START_TIME_UNIX_SEC = 0
BEST_TRACK_END_TIME_UNIX_SEC = 600
MAX_EXTRAP_TIME_LARGE_SECONDS = 200
MAX_EXTRAP_TIME_SMALL_SECONDS = 1
MAX_JOIN_TIME_LARGE_SECONDS = 301
MAX_JOIN_TIME_SMALL_SECONDS = 1

THESE_TRACKING_START_TIMES_UNIX_SEC = numpy.full(9, 0, dtype=int)
THESE_TRACKING_END_TIMES_UNIX_SEC = numpy.full(9, 600, dtype=int)
THESE_AGES_SECONDS = numpy.array(
    [-1, -1, -1, 0, -1, -1, -1, -1, -1], dtype=int)
THESE_CELL_START_TIMES_UNIX_SEC = numpy.array(
    [0, 0, 0, 300, 0, 0, 300, 300, 0], dtype=int)
THESE_CELL_END_TIMES_UNIX_SEC = numpy.array(
    [600, 300, 0, 300, 300, 600, 600, 600, 600], dtype=int)

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_STORM_IDS,
    tracking_utils.TIME_COLUMN: THESE_TIMES_UNIX_SEC
}
STORM_OBJECT_TABLE_NO_AGES = pandas.DataFrame.from_dict(THIS_DICT)

THIS_DICT = {
    tracking_utils.TRACKING_START_TIME_COLUMN:
        THESE_TRACKING_START_TIMES_UNIX_SEC,
    tracking_utils.TRACKING_END_TIME_COLUMN: THESE_TRACKING_END_TIMES_UNIX_SEC,
    tracking_utils.AGE_COLUMN: THESE_AGES_SECONDS,
    tracking_utils.CELL_START_TIME_COLUMN: THESE_CELL_START_TIMES_UNIX_SEC,
    tracking_utils.CELL_END_TIME_COLUMN: THESE_CELL_END_TIMES_UNIX_SEC
}
STORM_OBJECT_TABLE_SOME_VALID_AGES = copy.deepcopy(STORM_OBJECT_TABLE_NO_AGES)
STORM_OBJECT_TABLE_SOME_VALID_AGES = STORM_OBJECT_TABLE_SOME_VALID_AGES.assign(
    **THIS_DICT)

THIS_DICT = {
    tracking_utils.AGE_COLUMN: numpy.full(len(THESE_STORM_IDS), -1, dtype=int)
}
STORM_OBJECT_TABLE_INVALID_AGES = copy.deepcopy(
    STORM_OBJECT_TABLE_SOME_VALID_AGES)
STORM_OBJECT_TABLE_INVALID_AGES = (
    STORM_OBJECT_TABLE_INVALID_AGES.assign(**THIS_DICT))


def _compare_storm_track_tables(
        first_storm_track_table, second_storm_track_table):
    """Determines equality of two tables with storm-tracking data.

    :param first_storm_track_table: First table (pandas DataFrame).
    :param second_storm_track_table: Second table.
    :return: are_tables_equal: Boolean flag.
    """

    first_column_names = list(first_storm_track_table)
    second_column_names = list(second_storm_track_table)
    if set(first_column_names) != set(second_column_names):
        return False

    num_tracks = len(first_storm_track_table.index)
    for this_column in first_column_names:
        if this_column in STRING_COLUMNS_IN_TRACK_TABLE:
            if not numpy.array_equal(
                    first_storm_track_table[this_column].values,
                    second_storm_track_table[this_column].values):
                return False

        elif this_column in ARRAY_COLUMNS_IN_TRACK_TABLE:
            for i in range(num_tracks):
                if not numpy.allclose(
                        first_storm_track_table[this_column].values[i],
                        second_storm_track_table[this_column].values[i],
                        atol=TOLERANCE):
                    return False

        else:
            if not numpy.allclose(
                    first_storm_track_table[this_column].values,
                    second_storm_track_table[this_column].values,
                    atol=TOLERANCE):
                return False

    return True


class BestTracksTests(unittest.TestCase):
    """Each method is a unit test for best_tracks.py."""

    def test__theil_sen_fit_one_track(self):
        """Ensures correct output from _theil_sen_fit_one_track."""

        (this_x_intercept_metres, this_x_velocity_m_s01,
         this_y_intercept_metres, this_y_velocity_m_s01
        ) = best_tracks._theil_sen_fit_one_track(
            object_x_coords_metres=X_COORDS_ONE_TRACK_METRES,
            object_y_coords_metres=Y_COORDS_ONE_TRACK_METRES,
            object_times_unix_sec=VALID_TIMES_ONE_TRACK_UNIX_SEC)

        self.assertTrue(numpy.isclose(
            this_x_intercept_metres, X_INTERCEPT_ONE_TRACK_METRES,
            atol=TOLERANCE))
        self.assertTrue(numpy.isclose(
            this_x_velocity_m_s01, X_VELOCITY_ONE_TRACK_M_S01, atol=TOLERANCE))
        self.assertTrue(numpy.isclose(
            this_y_intercept_metres, Y_INTERCEPT_ONE_TRACK_METRES,
            atol=TOLERANCE))
        self.assertTrue(numpy.isclose(
            this_y_velocity_m_s01, Y_VELOCITY_ONE_TRACK_M_S01, atol=TOLERANCE))

    def test_theil_sen_predict(self):
        """Ensures correct output from _theil_sen_predict."""

        (this_x_predicted_metres, this_y_predicted_metres
        ) = best_tracks._theil_sen_predict(
            x_intercept_metres=X_INTERCEPT_ONE_TRACK_METRES,
            x_velocity_m_s01=X_VELOCITY_ONE_TRACK_M_S01,
            y_intercept_metres=Y_INTERCEPT_ONE_TRACK_METRES,
            y_velocity_m_s01=Y_VELOCITY_ONE_TRACK_M_S01,
            query_time_unix_sec=QUERY_TIMES_ONE_TRACK_UNIX_SEC[0])

        self.assertTrue(numpy.isclose(
            this_x_predicted_metres, QUERY_X_COORDS_ONE_TRACK_METRES[0],
            atol=TOLERANCE))
        self.assertTrue(numpy.isclose(
            this_y_predicted_metres, QUERY_Y_COORDS_ONE_TRACK_METRES[0],
            atol=TOLERANCE))

    def test_theil_sen_predict_many_times(self):
        """Ensures correct output from _theil_sen_predict_many_times."""

        (these_x_predicted_metres, these_y_predicted_metres
        ) = best_tracks._theil_sen_predict_many_times(
            x_intercept_metres=X_INTERCEPT_ONE_TRACK_METRES,
            x_velocity_m_s01=X_VELOCITY_ONE_TRACK_M_S01,
            y_intercept_metres=Y_INTERCEPT_ONE_TRACK_METRES,
            y_velocity_m_s01=Y_VELOCITY_ONE_TRACK_M_S01,
            query_times_unix_sec=QUERY_TIMES_ONE_TRACK_UNIX_SEC)

        self.assertTrue(numpy.allclose(
            these_x_predicted_metres, QUERY_X_COORDS_ONE_TRACK_METRES,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_y_predicted_metres, QUERY_Y_COORDS_ONE_TRACK_METRES,
            atol=TOLERANCE))

    def test_theil_sen_predict_many_models(self):
        """Ensures correct output from _theil_sen_predict_many_models."""

        (these_x_predicted_metres, these_y_predicted_metres
        ) = best_tracks._theil_sen_predict_many_models(
            x_intercepts_metres=X_INTERCEPTS_MANY_MODELS_METRES,
            x_velocities_m_s01=X_VELOCITIES_MANY_MODELS_M_S01,
            y_intercepts_metres=Y_INTERCEPTS_MANY_MODELS_METRES,
            y_velocities_m_s01=Y_VELOCITIES_MANY_MODELS_M_S01,
            query_time_unix_sec=QUERY_TIME_MANY_MODELS_UNIX_SEC)

        self.assertTrue(numpy.allclose(
            these_x_predicted_metres, QUERY_X_COORDS_MANY_MODELS_METRES,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_y_predicted_metres, QUERY_Y_COORDS_MANY_MODELS_METRES,
            atol=TOLERANCE))

    def test_get_theil_sen_error_one_track(self):
        """Ensures correct output for _get_theil_sen_error_one_track."""

        this_rmse_metres = best_tracks._get_theil_sen_error_one_track(
            storm_track_table=ONE_TRACK_TABLE_NOISY, storm_track_index=0)
        self.assertTrue(numpy.isclose(
            this_rmse_metres, ONE_TRACK_RMSE_METRES, atol=TOLERANCE))

    def test_get_theil_sen_errors_one_object(self):
        """Ensures correct output from _get_theil_sen_errors_one_object."""

        this_error_metres = best_tracks._get_theil_sen_errors_one_object(
            object_x_metres=ONE_TRACK_TABLE_NOISY[
                best_tracks.TRACK_X_COORDS_COLUMN].values[0][0],
            object_y_metres=ONE_TRACK_TABLE_NOISY[
                best_tracks.TRACK_Y_COORDS_COLUMN].values[0][0],
            object_time_unix_sec=ONE_TRACK_TABLE_NOISY[
                best_tracks.TRACK_TIMES_COLUMN].values[0][0],
            storm_track_table=ONE_TRACK_TABLE_NOISY)[0]

        self.assertTrue(numpy.isclose(
            this_error_metres, ONE_TRACK_FIRST_OBJECT_ERROR_METRES,
            atol=TOLERANCE))

    def test_get_join_time_for_two_tracks_overlapping(self):
        """Ensures correct output from _get_join_time_for_two_tracks.

        In this case the two tracks overlap in time.
        """

        (this_join_time_seconds, this_early_index, this_late_index
        ) = best_tracks._get_join_time_for_two_tracks(
            start_times_unix_sec=START_TIMES_OVERLAPPING_TRACKS_UNIX_SEC,
            end_times_unix_sec=END_TIMES_OVERLAPPING_TRACKS_UNIX_SEC)

        self.assertTrue(numpy.isnan(this_join_time_seconds))
        self.assertTrue(this_early_index == EARLY_INDEX_OVERLAPPING_TRACKS)
        self.assertTrue(this_late_index == LATE_INDEX_OVERLAPPING_TRACKS)

    def test_get_join_time_for_two_tracks_non_overlapping(self):
        """Ensures correct output from _get_join_time_for_two_tracks.

        In this case the two tracks do *not* overlap in time.
        """

        (this_join_time_seconds, this_early_index, this_late_index
        ) = best_tracks._get_join_time_for_two_tracks(
            start_times_unix_sec=START_TIMES_NON_OVERLAPPING_TRACKS_UNIX_SEC,
            end_times_unix_sec=END_TIMES_NON_OVERLAPPING_TRACKS_UNIX_SEC)

        self.assertTrue(this_join_time_seconds == JOIN_TIME_NON_OVERLAPPING_SEC)
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
            this_join_distance_metres, JOIN_DISTANCE_TWO_TRACKS_METRES,
            atol=TOLERANCE))

    def test_get_velocity_difference_two_tracks(self):
        """Ensures correct output from _get_velocity_difference_two_tracks."""

        this_storm_track_table = best_tracks.theil_sen_fit_many_tracks(
            storm_track_table=copy.deepcopy(TWO_STORM_TRACKS_TABLE),
            verbose=False)

        this_velocity_diff_m_s01 = (
            best_tracks._get_velocity_difference_two_tracks(
                storm_track_table=this_storm_track_table, first_index=0,
                second_index=1))

        self.assertTrue(numpy.isclose(
            this_velocity_diff_m_s01, VELOCITY_DIFF_TWO_TRACKS_M_S01,
            atol=TOLERANCE))

    def test_get_theil_sen_error_two_tracks(self):
        """Ensures correct output from _get_theil_sen_error_two_tracks."""

        this_storm_track_table = best_tracks.theil_sen_fit_many_tracks(
            storm_track_table=copy.deepcopy(TWO_STORM_TRACKS_TABLE),
            verbose=False)

        this_mean_error_m_s01 = best_tracks._get_theil_sen_error_two_tracks(
            storm_track_table=this_storm_track_table, early_track_index=1,
            late_track_index=0)

        self.assertTrue(numpy.isclose(
            this_mean_error_m_s01, MEAN_ERROR_TWO_TRACKS_M_S01, atol=TOLERANCE))

    def test_break_ties_one_track(self):
        """Ensures correct output from _break_ties_one_track."""

        this_storm_track_table = best_tracks.theil_sen_fit_many_tracks(
            storm_track_table=copy.deepcopy(STORM_TRACK_TABLE_FOR_TIEBREAKER),
            verbose=False)

        these_indices_to_remove = best_tracks._break_ties_one_track(
            storm_track_table=this_storm_track_table, storm_track_index=0)
        self.assertTrue(numpy.array_equal(
            these_indices_to_remove, INDICES_TO_REMOVE_FOR_TIEBREAKER))

    def test_find_changed_tracks(self):
        """Ensures correct output from _find_changed_tracks."""

        these_track_changed_indices = best_tracks._find_changed_tracks(
            storm_track_table=MAIN_STORM_TRACK_TABLE_TWEAKED,
            orig_storm_track_table=MAIN_STORM_TRACK_TABLE)
        self.assertTrue(numpy.array_equal(
            these_track_changed_indices, TRACK_CHANGED_INDICES))

    def test_storm_objects_to_tracks_all(self):
        """Ensures correct output from storm_objects_to_tracks.

        In this case, tracking data are created for all storms.
        """

        this_storm_track_table = best_tracks.storm_objects_to_tracks(
            MAIN_STORM_OBJECT_TABLE)
        this_storm_track_table.sort_values(
            tracking_utils.STORM_ID_COLUMN, axis=0, ascending=True,
            inplace=True)

        self.assertTrue(_compare_storm_track_tables(
            this_storm_track_table, MAIN_STORM_TRACK_TABLE))

    def test_storm_objects_to_tracks_foo_only(self):
        """Ensures correct output from storm_objects_to_tracks.

        In this case, tracking data are created only for storm "foo".
        """

        this_storm_track_table = best_tracks.storm_objects_to_tracks(
            MAIN_STORM_OBJECT_TABLE, storm_ids_to_use=['foo'])
        self.assertTrue(_compare_storm_track_tables(
            this_storm_track_table, STORM_TRACK_TABLE_FOO_ONLY))

    def test_remove_short_tracks_min_length_1(self):
        """Ensures correct output from remove_short_tracks.

        In this case, minimum track length is one storm object.
        """

        this_storm_object_table = best_tracks.remove_short_tracks(
            storm_object_table=MAIN_STORM_OBJECT_TABLE, min_objects_in_track=1)

        self.assertTrue(numpy.array_equal(
            this_storm_object_table[tracking_utils.STORM_ID_COLUMN].values,
            STORM_OBJECT_TABLE_TRACK_LENGTH_GEQ1[
                tracking_utils.STORM_ID_COLUMN
            ].values))

        self.assertTrue(numpy.array_equal(
            this_storm_object_table[tracking_utils.TIME_COLUMN].values,
            STORM_OBJECT_TABLE_TRACK_LENGTH_GEQ1[
                tracking_utils.TIME_COLUMN
            ].values))

    def test_remove_short_tracks_min_length_2(self):
        """Ensures correct output from remove_short_tracks.

        In this case, minimum track length is 2 storm objects.
        """

        this_storm_object_table = best_tracks.remove_short_tracks(
            storm_object_table=MAIN_STORM_OBJECT_TABLE, min_objects_in_track=2)

        self.assertTrue(numpy.array_equal(
            this_storm_object_table[tracking_utils.STORM_ID_COLUMN].values,
            STORM_OBJECT_TABLE_TRACK_LENGTH_GEQ2[
                tracking_utils.STORM_ID_COLUMN
            ].values))

        self.assertTrue(numpy.array_equal(
            this_storm_object_table[tracking_utils.TIME_COLUMN].values,
            STORM_OBJECT_TABLE_TRACK_LENGTH_GEQ2[
                tracking_utils.TIME_COLUMN
            ].values))

    def test_remove_short_tracks_min_length_3(self):
        """Ensures correct output from remove_short_tracks.

        In this case, minimum track length is 3 storm objects.
        """

        this_storm_object_table = best_tracks.remove_short_tracks(
            storm_object_table=MAIN_STORM_OBJECT_TABLE, min_objects_in_track=3)

        self.assertTrue(numpy.array_equal(
            this_storm_object_table[tracking_utils.STORM_ID_COLUMN].values,
            STORM_OBJECT_TABLE_TRACK_LENGTH_GEQ3[
                tracking_utils.STORM_ID_COLUMN
            ].values))

        self.assertTrue(numpy.array_equal(
            this_storm_object_table[tracking_utils.TIME_COLUMN].values,
            STORM_OBJECT_TABLE_TRACK_LENGTH_GEQ3[
                tracking_utils.TIME_COLUMN
            ].values))

    def test_get_storm_ages_all_invalid(self):
        """Ensures correct output from get_storm_ages.

        In this case all storm ages are invalid ("-1 seconds"), because all
        storms either start too near the beginning of the tracking period or end
        too near the end of the period.
        """

        this_storm_object_table = best_tracks.get_storm_ages(
            storm_object_table=copy.deepcopy(STORM_OBJECT_TABLE_NO_AGES),
            best_track_start_time_unix_sec=BEST_TRACK_START_TIME_UNIX_SEC,
            best_track_end_time_unix_sec=BEST_TRACK_END_TIME_UNIX_SEC,
            max_extrap_time_for_breakup_sec=MAX_EXTRAP_TIME_LARGE_SECONDS,
            max_join_time_sec=MAX_JOIN_TIME_LARGE_SECONDS)

        self.assertTrue(
            this_storm_object_table.equals(STORM_OBJECT_TABLE_INVALID_AGES))

    def test_get_storm_ages_some_valid(self):
        """Ensures correct output from get_storm_ages.

        In this case some storm ages are valid (not "-1 seconds").
        """

        this_storm_object_table = best_tracks.get_storm_ages(
            storm_object_table=copy.deepcopy(STORM_OBJECT_TABLE_NO_AGES),
            best_track_start_time_unix_sec=BEST_TRACK_START_TIME_UNIX_SEC,
            best_track_end_time_unix_sec=BEST_TRACK_END_TIME_UNIX_SEC,
            max_extrap_time_for_breakup_sec=MAX_EXTRAP_TIME_SMALL_SECONDS,
            max_join_time_sec=MAX_JOIN_TIME_SMALL_SECONDS)

        self.assertTrue(
            this_storm_object_table.equals(STORM_OBJECT_TABLE_SOME_VALID_AGES))


if __name__ == '__main__':
    unittest.main()
