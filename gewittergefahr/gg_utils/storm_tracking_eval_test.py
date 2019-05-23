"""Unit tests for storm_tracking_eval.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import storm_tracking_eval as tracking_eval
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

TOLERANCE = 1e-6

# The following constants are used to test _fit_theil_sen_one_track.
VALID_TIMES_ONE_TRACK_UNIX_SEC = numpy.array(
    [0, 1, 2, 2, 3, 4, 5, 6, 7, 7], dtype=int
)
X_COORDS_ONE_TRACK_METRES = numpy.array([
    -10, -7.5, -5, -5, -2.5, 0, 2.5, 5, 7.5, 7.5
])
Y_COORDS_ONE_TRACK_METRES = numpy.array([
    5, 1.4, -2.2, -2.2, -5.8, -9.4, -13, -16.6, -20.2, -20.2
])

X_INTERCEPT_ONE_TRACK_METRES = -10.
X_VELOCITY_ONE_TRACK_M_S01 = 2.5
Y_INTERCEPT_ONE_TRACK_METRES = 5.
Y_VELOCITY_ONE_TRACK_M_S01 = -3.6

# The following constants are used to test _apply_theil_sen_one_track.
STORM_TRACK_DICT = {
    tracking_eval.X_INTERCEPT_COLUMN: [X_INTERCEPT_ONE_TRACK_METRES],
    tracking_eval.X_VELOCITY_COLUMN: [X_VELOCITY_ONE_TRACK_M_S01],
    tracking_eval.Y_INTERCEPT_COLUMN: [Y_INTERCEPT_ONE_TRACK_METRES],
    tracking_eval.Y_VELOCITY_COLUMN: [Y_VELOCITY_ONE_TRACK_M_S01]
}

STORM_TRACK_TABLE = pandas.DataFrame.from_dict(STORM_TRACK_DICT)

NESTED_ARRAY = STORM_TRACK_TABLE[[
    tracking_eval.X_INTERCEPT_COLUMN, tracking_eval.X_INTERCEPT_COLUMN
]].values.tolist()

STORM_TRACK_TABLE = STORM_TRACK_TABLE.assign(**{
    tracking_utils.TRACK_TIMES_COLUMN: NESTED_ARRAY,
    tracking_utils.TRACK_X_COORDS_COLUMN: NESTED_ARRAY,
    tracking_utils.TRACK_Y_COORDS_COLUMN: NESTED_ARRAY
})

THESE_TIMES_UNIX_SEC = numpy.array([10, 10, 15, 15, 20, 25, 30, 40], dtype=int)
ACTUAL_X_COORDS_METRES = numpy.array([12, 19, 22, 33, 38, 50.5, 63, 86])
ACTUAL_Y_COORDS_METRES = numpy.array([
    -38, -25, -60, -45, -66.6, -88, -102, -130
])

EXPECTED_X_COORDS_METRES = numpy.array([15, 15, 27.5, 27.5, 40, 52.5, 65, 90])
EXPECTED_Y_COORDS_METRES = numpy.array(
    [-31, -31, -49, -49, -67, -85, -103, -139], dtype=float
)

STORM_TRACK_TABLE[tracking_utils.TRACK_TIMES_COLUMN].values[0] = (
    THESE_TIMES_UNIX_SEC
)
STORM_TRACK_TABLE[tracking_utils.TRACK_X_COORDS_COLUMN].values[0] = (
    ACTUAL_X_COORDS_METRES
)
STORM_TRACK_TABLE[tracking_utils.TRACK_Y_COORDS_COLUMN].values[0] = (
    ACTUAL_Y_COORDS_METRES
)

# The following constants are used to test _get_mean_ts_error_one_track.
RMSE_METRES = numpy.sqrt(numpy.mean(
    (EXPECTED_X_COORDS_METRES - ACTUAL_X_COORDS_METRES) ** 2 +
    (EXPECTED_Y_COORDS_METRES - ACTUAL_Y_COORDS_METRES) ** 2
))


class StormTrackingEvalTests(unittest.TestCase):
    """Each method is a unit test for storm_tracking_eval.py."""

    def test_fit_theil_sen_one_track(self):
        """Ensures correct output from _fit_theil_sen_one_track."""

        this_dict = tracking_eval._fit_theil_sen_one_track(
            x_coords_metres=X_COORDS_ONE_TRACK_METRES,
            y_coords_metres=Y_COORDS_ONE_TRACK_METRES,
            valid_times_unix_sec=VALID_TIMES_ONE_TRACK_UNIX_SEC)

        self.assertTrue(numpy.isclose(
            this_dict[tracking_eval.X_INTERCEPT_KEY],
            X_INTERCEPT_ONE_TRACK_METRES, atol=TOLERANCE
        ))

        self.assertTrue(numpy.isclose(
            this_dict[tracking_eval.X_VELOCITY_KEY],
            X_VELOCITY_ONE_TRACK_M_S01, atol=TOLERANCE
        ))

        self.assertTrue(numpy.isclose(
            this_dict[tracking_eval.Y_INTERCEPT_KEY],
            Y_INTERCEPT_ONE_TRACK_METRES, atol=TOLERANCE
        ))

        self.assertTrue(numpy.isclose(
            this_dict[tracking_eval.Y_VELOCITY_KEY],
            Y_VELOCITY_ONE_TRACK_M_S01, atol=TOLERANCE
        ))

    def test_apply_theil_sen_one_track(self):
        """Ensures correct output from _apply_theil_sen_one_track."""

        these_x_coords_metres, these_y_coords_metres = (
            tracking_eval._apply_theil_sen_one_track(
                storm_track_table=copy.deepcopy(STORM_TRACK_TABLE), row_index=0)
        )

        self.assertTrue(numpy.allclose(
            these_x_coords_metres, EXPECTED_X_COORDS_METRES, atol=TOLERANCE
        ))

        self.assertTrue(numpy.allclose(
            these_y_coords_metres, EXPECTED_Y_COORDS_METRES, atol=TOLERANCE
        ))

    def test_get_mean_ts_error_one_track(self):
        """Ensures correct output from _get_mean_ts_error_one_track."""

        this_rmse_metres = tracking_eval._get_mean_ts_error_one_track(
            storm_track_table=copy.deepcopy(STORM_TRACK_TABLE), row_index=0
        )

        self.assertTrue(numpy.isclose(
            RMSE_METRES, this_rmse_metres, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
