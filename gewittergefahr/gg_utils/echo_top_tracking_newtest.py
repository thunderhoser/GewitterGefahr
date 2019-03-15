"""Unit tests for echo_top_tracking.py."""

import copy
import unittest
import numpy
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import time_conversion

TOLERANCE = 1e-6
VELOCITY_EFOLD_RADIUS_METRES = 1.

# The following constants are used to test _get_intermediate_velocities.
THESE_X_COORDS_METRES = numpy.array([2, -7, 1, 6, 5, -4], dtype=float)
THESE_Y_COORDS_METRES = numpy.array([4, -1, 5, -1, -3, 9], dtype=float)

FIRST_MAX_DICT_NO_VELOCITY = {
    echo_top_tracking.VALID_TIME_KEY: 0,
    echo_top_tracking.X_COORDS_KEY: THESE_X_COORDS_METRES + 0.,
    echo_top_tracking.Y_COORDS_KEY: THESE_Y_COORDS_METRES + 0.,
    echo_top_tracking.LONGITUDES_KEY: THESE_X_COORDS_METRES + 0.,
    echo_top_tracking.LATITUDES_KEY: THESE_Y_COORDS_METRES + 0.,
}

THESE_X_COORDS_METRES = numpy.array(
    [13, -1, 20, 20, -8, 5, -23, 19], dtype=float)
THESE_Y_COORDS_METRES = numpy.array(
    [-14, 25, -12, 1, -14, 4, -5, 18], dtype=float)

THIS_CURRENT_TO_PREV_MATRIX = numpy.array(
    [[0, 0, 0, 0, 0, 0],
     [1, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1]], dtype=bool)

SECOND_MAX_DICT_NO_VELOCITY = {
    echo_top_tracking.VALID_TIME_KEY: 10,
    echo_top_tracking.X_COORDS_KEY: THESE_X_COORDS_METRES + 0.,
    echo_top_tracking.Y_COORDS_KEY: THESE_Y_COORDS_METRES + 0.,
    echo_top_tracking.LONGITUDES_KEY: THESE_X_COORDS_METRES + 0.,
    echo_top_tracking.LATITUDES_KEY: THESE_Y_COORDS_METRES + 0.,
    echo_top_tracking.CURRENT_TO_PREV_MATRIX_KEY:
        copy.deepcopy(THIS_CURRENT_TO_PREV_MATRIX)
}

FIRST_MAX_DICT_WITH_VELOCITY = copy.deepcopy(FIRST_MAX_DICT_NO_VELOCITY)
FIRST_MAX_DICT_WITH_VELOCITY.update({
    echo_top_tracking.X_VELOCITIES_KEY: numpy.full(6, numpy.nan),
    echo_top_tracking.Y_VELOCITIES_KEY: numpy.full(6, numpy.nan)
})

SECOND_MAX_DICT_WITH_VELOCITY = copy.deepcopy(SECOND_MAX_DICT_NO_VELOCITY)

THESE_X_VELOCITIES_M_S01 = numpy.array(
    [numpy.nan, (-0.3 + 0.6) / 2, 1.9, numpy.nan, -0.9, 0, numpy.nan, 2.3])
THESE_Y_VELOCITIES_M_S01 = numpy.array(
    [numpy.nan, (2.1 + 2.6) / 2, -1.7, numpy.nan, -1.9, 0.7, numpy.nan, 0.9])

SECOND_MAX_DICT_WITH_VELOCITY.update({
    echo_top_tracking.X_VELOCITIES_KEY: THESE_X_VELOCITIES_M_S01,
    echo_top_tracking.Y_VELOCITIES_KEY: THESE_Y_VELOCITIES_M_S01
})

# The following constants are used to test _link_local_maxima_by_velocity,
# _link_local_maxima_by_distance, _prune_connections, and
# _link_local_maxima_in_time.
MAX_LINK_TIME_SECONDS = 100
MAX_VELOCITY_DIFF_M_S01 = 3.
MAX_LINK_DISTANCE_M_S01 = 2.

FIRST_LOCAL_MAX_DICT_UNLINKED = copy.deepcopy(FIRST_MAX_DICT_WITH_VELOCITY)
SECOND_LOCAL_MAX_DICT_UNLINKED = copy.deepcopy(SECOND_MAX_DICT_WITH_VELOCITY)
SECOND_LOCAL_MAX_DICT_UNLINKED.pop(echo_top_tracking.CURRENT_TO_PREV_MATRIX_KEY)

NUM_FIRST_MAXIMA = len(
    FIRST_LOCAL_MAX_DICT_UNLINKED[echo_top_tracking.X_COORDS_KEY]
)
NUM_SECOND_MAXIMA = len(
    SECOND_LOCAL_MAX_DICT_UNLINKED[echo_top_tracking.X_COORDS_KEY]
)

VELOCITY_DIFF_MATRIX_1TO2_M_S01 = numpy.full(
    (NUM_SECOND_MAXIMA, NUM_FIRST_MAXIMA), numpy.inf)
CURRENT_TO_PREV_MATRIX_VELOCITY_1TO2 = numpy.full(
    (NUM_SECOND_MAXIMA, NUM_FIRST_MAXIMA), False, dtype=bool)

THIS_X_DISTANCE_MATRIX_METRES = numpy.array([
    [11, 20, 12, 7, 8, 17],
    [3, 6, 2, 7, 6, 3],
    [18, 27, 19, 14, 15, 24],
    [18, 27, 19, 14, 15, 24],
    [10, 1, 9, 14, 13, 4],
    [3, 12, 4, 1, 0, 9],
    [25, 16, 24, 29, 28, 19],
    [17, 26, 18, 13, 14, 23]
], dtype=float)

THIS_Y_DISTANCE_MATRIX_METRES = numpy.array([
    [18, 13, 19, 13, 11, 23],
    [21, 26, 20, 26, 28, 16],
    [16, 11, 17, 11, 9, 21],
    [3, 2, 4, 2, 4, 8],
    [18, 13, 19, 13, 11, 23],
    [0, 5, 1, 5, 7, 5],
    [9, 4, 10, 4, 2, 14],
    [14, 19, 13, 19, 21, 9]
], dtype=float)

DISTANCE_MATRIX_1TO2_M_S01 = numpy.sqrt(
    THIS_X_DISTANCE_MATRIX_METRES ** 2 + THIS_Y_DISTANCE_MATRIX_METRES ** 2
) / 10

CURRENT_TO_PREV_MATRIX_DISTANCE_1TO2 = numpy.array(
    [[0, 0, 0, 1, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 1, 1, 0],
     [0, 0, 0, 1, 1, 0],
     [0, 1, 0, 0, 1, 0],
     [1, 0, 1, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0]], dtype=bool)

CURRENT_TO_PREV_MATRIX_1TO2 = numpy.array(
    [[0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [1, 0, 1, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0]], dtype=bool)

THESE_X_COORDS_METRES = numpy.array(
    [21, 5, 30, 31, 0, 17, 28], dtype=float)
THESE_Y_COORDS_METRES = numpy.array(
    [-9, 34, 0, 12, -4, 12, 25], dtype=float)

THIRD_LOCAL_MAX_DICT_UNLINKED = {
    echo_top_tracking.VALID_TIME_KEY: 15,
    echo_top_tracking.X_COORDS_KEY: THESE_X_COORDS_METRES + 0.,
    echo_top_tracking.Y_COORDS_KEY: THESE_Y_COORDS_METRES + 0.,
}

# SECOND_EXTRAP_X_COORDS_METRES = numpy.array(
#     [numpy.nan, -0.25, 29.5, numpy.nan, -12.5, 5, numpy.nan, 30.5])
# SECOND_EXTRAP_Y_COORDS_METRES = numpy.array(
#     [numpy.nan, 36.75, -20.5, numpy.nan, -23.5, 7.5, numpy.nan, 22.5])

THIS_X_DISTANCE_MATRIX_METRES = numpy.array(
    [[-1, 21.25, 8.5, -1, 33.5, 16, -1, 9.5],
     [-1, 5.25, 24.5, -1, 17.5, 0, -1, 25.5],
     [-1, 30.25, 0.5, -1, 42.5, 25, -1, 0.5],
     [-1, 31.25, 1.5, -1, 43.5, 26, -1, 0.5],
     [-1, 0.25, 29.5, -1, 12.5, 5, -1, 30.5],
     [-1, 17.25, 12.5, -1, 29.5, 12, -1, 13.5],
     [-1, 28.25, 1.5, -1, 40.5, 23, -1, 2.5]]
)

THIS_X_DISTANCE_MATRIX_METRES[THIS_X_DISTANCE_MATRIX_METRES < 0] = numpy.inf

THIS_Y_DISTANCE_MATRIX_METRES = numpy.array(
    [[-1, 45.75, 11.5, -1, 14.5, 16.5, -1, 31.5],
     [-1, 2.75, 54.5, -1, 57.5, 26.5, -1, 11.5],
     [-1, 36.75, 20.5, -1, 23.5, 7.5, -1, 22.5],
     [-1, 24.75, 32.5, -1, 35.5, 4.5, -1, 10.5],
     [-1, 40.75, 16.5, -1, 19.5, 11.5, -1, 26.5],
     [-1, 24.75, 32.5, -1, 35.5, 4.5, -1, 10.5],
     [-1, 11.75, 45.5, -1, 48.5, 17.5, -1, 2.5]]
)

THIS_Y_DISTANCE_MATRIX_METRES[THIS_Y_DISTANCE_MATRIX_METRES < 0] = numpy.inf

VELOCITY_DIFF_MATRIX_2TO3_M_S01 = numpy.sqrt(
    THIS_X_DISTANCE_MATRIX_METRES ** 2 + THIS_Y_DISTANCE_MATRIX_METRES ** 2
) / 5

CURRENT_TO_PREV_MATRIX_VELOCITY_2TO3 = numpy.array(
    [[0, 0, 1, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1]], dtype=bool)

THIS_X_DISTANCE_MATRIX_METRES = numpy.array([
    [8, -1, -1, 1, -1, -1, 44, -1],
    [8, -1, -1, 15, -1, -1, 28, -1],
    [17, -1, -1, 10, -1, -1, 53, -1],
    [18, -1, -1, 11, -1, -1, 54, -1],
    [13, -1, -1, 20, -1, -1, 23, -1],
    [4, -1, -1, 3, -1, -1, 40, -1],
    [15, -1, -1, 8, -1, -1, 51, -1]
], dtype=float)

THIS_X_DISTANCE_MATRIX_METRES[THIS_X_DISTANCE_MATRIX_METRES < 0] = numpy.inf

THIS_Y_DISTANCE_MATRIX_METRES = numpy.array([
    [5, -1, -1, 10, -1, -1, 4, -1],
    [48, -1, -1, 33, -1, -1, 39, -1],
    [14, -1, -1, 1, -1, -1, 5, -1],
    [26, -1, -1, 11, -1, -1, 17, -1],
    [10, -1, -1, 5, -1, -1, 1, -1],
    [26, -1, -1, 11, -1, -1, 17, -1],
    [39, -1, -1, 24, -1, -1, 30, -1]
], dtype=float)

THIS_Y_DISTANCE_MATRIX_METRES[THIS_Y_DISTANCE_MATRIX_METRES < 0] = numpy.inf

DISTANCE_MATRIX_2TO3_M_S01 = numpy.sqrt(
    THIS_X_DISTANCE_MATRIX_METRES ** 2 + THIS_Y_DISTANCE_MATRIX_METRES ** 2
) / 5

CURRENT_TO_PREV_MATRIX_DISTANCE_2TO3 = numpy.array(
    [[1, 0, 1, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1]], dtype=bool)

CURRENT_TO_PREV_MATRIX_2TO3 = numpy.array(
    [[1, 0, 1, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1]], dtype=bool)

# The following constants are used to test _create_primary_storm_id,
# _create_secondary_storm_id, and _create_full_storm_id.
PREV_SPC_DATE_STRING = '20190314'
PREV_PRIMARY_ID_NUMERIC = 16
PREV_SECONDARY_ID_NUMERIC = 34

NEXT_SPC_DATE_STRING = '20190315'
PRIMARY_ID_STRING_SAME_DAY = '000017_20190314'
PRIMARY_ID_STRING_NEXT_DAY = '000000_20190315'
SECONDARY_ID_STRING = '000035'
FULL_ID_STRING_SAME_DAY = '000017_20190314_000035'
FULL_ID_STRING_NEXT_DAY = '000000_20190315_000035'

# The following constants are used to test FOO.
FIRST_PRIMARY_ID_STRINGS = [
    '000000_19691231', '000001_19691231', '000002_19691231', '000003_19691231',
    '000004_19691231', '000005_19691231'
]
FIRST_SECONDARY_ID_STRINGS = [
    '000000', '000001', '000002', '000003', '000004', '000005'
]

FIRST_LOCAL_MAX_DICT_LINKED = copy.deepcopy(FIRST_LOCAL_MAX_DICT_UNLINKED)
FIRST_LOCAL_MAX_DICT_LINKED.update({
    echo_top_tracking.PRIMARY_IDS_KEY: FIRST_PRIMARY_ID_STRINGS,
    echo_top_tracking.SECONDARY_IDS_KEY: FIRST_SECONDARY_ID_STRINGS
})

PREV_SPC_DATE_STRING_1TO2_PREMERGE = '19691231'
PREV_PRIMARY_ID_1TO2_PREMERGE = 5
PREV_SECONDARY_ID_1TO2_PREMERGE = 5

SECOND_PRIMARY_ID_STRINGS_POSTMERGE = [
    '', '', '', '', '', '000006_19691231', '', ''
]
SECOND_SECONDARY_ID_STRINGS_POSTMERGE = [
    '', '', '', '', '', '000006', '', ''
]

PREV_SPC_DATE_STRING_1TO2_POSTMERGE = '19691231'
PREV_PRIMARY_ID_1TO2_POSTMERGE = 6
PREV_SECONDARY_ID_1TO2_POSTMERGE = 6

CURRENT_TO_PREV_MATRIX_1TO2_POSTMERGE = numpy.array(
    [[0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0]], dtype=bool)

SECOND_PRIMARY_ID_STRINGS_POSTSPLIT = [
    '000004_19691231', '', '', '', '000004_19691231', '000006_19691231', '', ''
]
SECOND_SECONDARY_ID_STRINGS_POSTSPLIT = [
    '000007', '', '', '', '000008', '000006', '', ''
]

PREV_SPC_DATE_STRING_1TO2_POSTSPLIT = '19691231'
PREV_PRIMARY_ID_1TO2_POSTSPLIT = 6
PREV_SECONDARY_ID_1TO2_POSTSPLIT = 8

CURRENT_TO_PREV_MATRIX_1TO2_POSTSPLIT = numpy.array(
    [[0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0]], dtype=bool)

SECOND_PRIMARY_ID_STRINGS = [
    '000004_19691231', '000005_19691231', '000007_19691231', '000003_19691231',
    '000004_19691231', '000006_19691231', '000001_19691231', '000008_19691231'
]
SECOND_SECONDARY_ID_STRINGS = [
    '000007', '000005', '000009', '000003', '000008', '000006', '000001',
    '000010'
]

PREV_SPC_DATE_STRING_1TO2_POST = '19691231'
PREV_PRIMARY_ID_1TO2_POST = 8
PREV_SECONDARY_ID_1TO2_POST = 10

SECOND_LOCAL_MAX_DICT_LINKED = copy.deepcopy(SECOND_LOCAL_MAX_DICT_UNLINKED)
SECOND_LOCAL_MAX_DICT_LINKED.update({
    echo_top_tracking.CURRENT_TO_PREV_MATRIX_KEY: CURRENT_TO_PREV_MATRIX_1TO2,
    echo_top_tracking.PRIMARY_IDS_KEY: SECOND_PRIMARY_ID_STRINGS,
    echo_top_tracking.SECONDARY_IDS_KEY: SECOND_SECONDARY_ID_STRINGS
})


class EchoTopTrackingTests(unittest.TestCase):
    """Each method is a unit test for echo_top_tracking.py."""

    def test_get_intermediate_velocities_time1(self):
        """Ensures correct output from _get_intermediate_velocities.

        In this case, "current time" = first time.
        """

        this_local_max_dict = echo_top_tracking._get_intermediate_velocities(
            current_local_max_dict=copy.deepcopy(FIRST_MAX_DICT_NO_VELOCITY),
            previous_local_max_dict=None, e_folding_radius_metres=0.)

        self.assertTrue(numpy.allclose(
            this_local_max_dict[echo_top_tracking.X_VELOCITIES_KEY],
            FIRST_MAX_DICT_WITH_VELOCITY[echo_top_tracking.X_VELOCITIES_KEY],
            atol=TOLERANCE, equal_nan=True
        ))

        self.assertTrue(numpy.allclose(
            this_local_max_dict[echo_top_tracking.Y_VELOCITIES_KEY],
            FIRST_MAX_DICT_WITH_VELOCITY[echo_top_tracking.Y_VELOCITIES_KEY],
            atol=TOLERANCE, equal_nan=True
        ))

    def test_get_intermediate_velocities_time2(self):
        """Ensures correct output from _get_intermediate_velocities.

        In this case, "current time" = second time.
        """

        this_local_max_dict = echo_top_tracking._get_intermediate_velocities(
            current_local_max_dict=copy.deepcopy(SECOND_MAX_DICT_NO_VELOCITY),
            previous_local_max_dict=copy.deepcopy(FIRST_MAX_DICT_NO_VELOCITY),
            e_folding_radius_metres=0.)

        self.assertTrue(numpy.allclose(
            this_local_max_dict[echo_top_tracking.X_VELOCITIES_KEY],
            SECOND_MAX_DICT_WITH_VELOCITY[echo_top_tracking.X_VELOCITIES_KEY],
            atol=TOLERANCE, equal_nan=True
        ))

        self.assertTrue(numpy.allclose(
            this_local_max_dict[echo_top_tracking.Y_VELOCITIES_KEY],
            SECOND_MAX_DICT_WITH_VELOCITY[echo_top_tracking.Y_VELOCITIES_KEY],
            atol=TOLERANCE, equal_nan=True
        ))

    def test_link_local_maxima_by_velocity_1to2(self):
        """Ensures correct output from _link_local_maxima_by_velocity.

        In this case, linking maxima from the first and second times.
        """

        this_diff_matrix_m_s01, this_current_to_prev_matrix = (
            echo_top_tracking._link_local_maxima_by_velocity(
                current_local_max_dict=SECOND_LOCAL_MAX_DICT_UNLINKED,
                previous_local_max_dict=FIRST_LOCAL_MAX_DICT_UNLINKED,
                max_velocity_diff_m_s01=MAX_VELOCITY_DIFF_M_S01)
        )

        self.assertTrue(numpy.allclose(
            this_diff_matrix_m_s01, VELOCITY_DIFF_MATRIX_1TO2_M_S01,
            atol=TOLERANCE
        ))

        self.assertTrue(numpy.array_equal(
            this_current_to_prev_matrix, CURRENT_TO_PREV_MATRIX_VELOCITY_1TO2
        ))

    def test_link_local_maxima_by_velocity_2to3(self):
        """Ensures correct output from _link_local_maxima_by_velocity.

        In this case, linking maxima from the second and third times.
        """

        this_diff_matrix_m_s01, this_current_to_prev_matrix = (
            echo_top_tracking._link_local_maxima_by_velocity(
                current_local_max_dict=THIRD_LOCAL_MAX_DICT_UNLINKED,
                previous_local_max_dict=SECOND_LOCAL_MAX_DICT_UNLINKED,
                max_velocity_diff_m_s01=MAX_VELOCITY_DIFF_M_S01)
        )

        self.assertTrue(numpy.allclose(
            this_diff_matrix_m_s01, VELOCITY_DIFF_MATRIX_2TO3_M_S01,
            atol=TOLERANCE
        ))

        self.assertTrue(numpy.array_equal(
            this_current_to_prev_matrix, CURRENT_TO_PREV_MATRIX_VELOCITY_2TO3
        ))

    def test_link_local_maxima_by_distance_1to2(self):
        """Ensures correct output from _link_local_maxima_by_distance.

        In this case, linking maxima from the first and second times.
        """

        this_diff_matrix_m_s01, this_current_to_prev_matrix = (
            echo_top_tracking._link_local_maxima_by_distance(
                current_local_max_dict=SECOND_LOCAL_MAX_DICT_UNLINKED,
                previous_local_max_dict=FIRST_LOCAL_MAX_DICT_UNLINKED,
                max_link_distance_m_s01=MAX_LINK_DISTANCE_M_S01,
                current_to_previous_matrix=copy.deepcopy(
                    CURRENT_TO_PREV_MATRIX_VELOCITY_1TO2)
            )
        )

        self.assertTrue(numpy.allclose(
            this_diff_matrix_m_s01, DISTANCE_MATRIX_1TO2_M_S01, atol=TOLERANCE
        ))

        self.assertTrue(numpy.array_equal(
            this_current_to_prev_matrix, CURRENT_TO_PREV_MATRIX_DISTANCE_1TO2
        ))

    def test_link_local_maxima_by_distance_2to3(self):
        """Ensures correct output from _link_local_maxima_by_distance.

        In this case, linking maxima from the second and third times.
        """

        this_diff_matrix_m_s01, this_current_to_prev_matrix = (
            echo_top_tracking._link_local_maxima_by_distance(
                current_local_max_dict=THIRD_LOCAL_MAX_DICT_UNLINKED,
                previous_local_max_dict=SECOND_LOCAL_MAX_DICT_UNLINKED,
                max_link_distance_m_s01=MAX_LINK_DISTANCE_M_S01,
                current_to_previous_matrix=copy.deepcopy(
                    CURRENT_TO_PREV_MATRIX_VELOCITY_2TO3)
            )
        )

        self.assertTrue(numpy.allclose(
            this_diff_matrix_m_s01, DISTANCE_MATRIX_2TO3_M_S01, atol=TOLERANCE
        ))

        self.assertTrue(numpy.array_equal(
            this_current_to_prev_matrix, CURRENT_TO_PREV_MATRIX_DISTANCE_2TO3
        ))

    def test_prune_connections_1to2(self):
        """Ensures correct output from _prune_connections.

        In this case, linking maxima from the first and second times.
        """

        this_current_to_prev_matrix = echo_top_tracking._prune_connections(
            velocity_diff_matrix_m_s01=VELOCITY_DIFF_MATRIX_1TO2_M_S01,
            distance_matrix_m_s01=DISTANCE_MATRIX_1TO2_M_S01,
            current_to_previous_matrix=copy.deepcopy(
                CURRENT_TO_PREV_MATRIX_DISTANCE_1TO2)
        )

        self.assertTrue(numpy.array_equal(
            this_current_to_prev_matrix, CURRENT_TO_PREV_MATRIX_1TO2
        ))

    def test_prune_connections_2to3(self):
        """Ensures correct output from _prune_connections.

        In this case, linking maxima from the second and third times.
        """

        this_current_to_prev_matrix = echo_top_tracking._prune_connections(
            velocity_diff_matrix_m_s01=VELOCITY_DIFF_MATRIX_2TO3_M_S01,
            distance_matrix_m_s01=DISTANCE_MATRIX_2TO3_M_S01,
            current_to_previous_matrix=copy.deepcopy(
                CURRENT_TO_PREV_MATRIX_DISTANCE_2TO3)
        )

        self.assertTrue(numpy.array_equal(
            this_current_to_prev_matrix, CURRENT_TO_PREV_MATRIX_2TO3
        ))

    def test_link_local_maxima_in_time_1to2(self):
        """Ensures correct output from _link_local_maxima_in_time.

        In this case, linking maxima from the first and second times.
        """

        this_current_to_prev_matrix = (
            echo_top_tracking._link_local_maxima_in_time(
                current_local_max_dict=SECOND_LOCAL_MAX_DICT_UNLINKED,
                previous_local_max_dict=FIRST_LOCAL_MAX_DICT_UNLINKED,
                max_link_time_seconds=MAX_LINK_TIME_SECONDS,
                max_velocity_diff_m_s01=MAX_VELOCITY_DIFF_M_S01,
                max_link_distance_m_s01=MAX_LINK_DISTANCE_M_S01)
        )

        self.assertTrue(numpy.array_equal(
            this_current_to_prev_matrix, CURRENT_TO_PREV_MATRIX_1TO2
        ))

    def test_link_local_maxima_in_time_2to3(self):
        """Ensures correct output from _link_local_maxima_in_time.

        In this case, linking maxima from the second and third times.
        """

        this_current_to_prev_matrix = (
            echo_top_tracking._link_local_maxima_in_time(
                current_local_max_dict=THIRD_LOCAL_MAX_DICT_UNLINKED,
                previous_local_max_dict=SECOND_LOCAL_MAX_DICT_UNLINKED,
                max_link_time_seconds=MAX_LINK_TIME_SECONDS,
                max_velocity_diff_m_s01=MAX_VELOCITY_DIFF_M_S01,
                max_link_distance_m_s01=MAX_LINK_DISTANCE_M_S01)
        )

        self.assertTrue(numpy.array_equal(
            this_current_to_prev_matrix, CURRENT_TO_PREV_MATRIX_2TO3
        ))

    def test_create_primary_storm_id_same_day(self):
        """Ensures correct output from _create_primary_storm_id.

        In this case the new storm occurs on the same SPC date as the previous
        storm to get a new ID.
        """

        this_id_string, this_id_numeric, this_spc_date_string = (
            echo_top_tracking._create_primary_storm_id(
                storm_start_time_unix_sec=time_conversion.get_start_of_spc_date(
                    PREV_SPC_DATE_STRING),
                previous_numeric_id=PREV_PRIMARY_ID_NUMERIC,
                previous_spc_date_string=PREV_SPC_DATE_STRING)
        )

        self.assertTrue(this_id_string == PRIMARY_ID_STRING_SAME_DAY)
        self.assertTrue(this_id_numeric == PREV_PRIMARY_ID_NUMERIC + 1)
        self.assertTrue(this_spc_date_string == PREV_SPC_DATE_STRING)

    def test_create_primary_storm_id_next_day(self):
        """Ensures correct output from _create_primary_storm_id.

        In this case the new storm *does not* occur on the same SPC date as the
        previous storm to get a new ID.
        """

        this_id_string, this_id_numeric, this_spc_date_string = (
            echo_top_tracking._create_primary_storm_id(
                storm_start_time_unix_sec=time_conversion.get_start_of_spc_date(
                    NEXT_SPC_DATE_STRING),
                previous_numeric_id=PREV_PRIMARY_ID_NUMERIC,
                previous_spc_date_string=PREV_SPC_DATE_STRING)
        )

        self.assertTrue(this_id_string == PRIMARY_ID_STRING_NEXT_DAY)
        self.assertTrue(this_id_numeric == 0)
        self.assertTrue(this_spc_date_string == NEXT_SPC_DATE_STRING)

    def test_create_secondary_storm_id(self):
        """Ensures correct output from _create_secondary_storm_id."""

        this_id_string, this_id_numeric = (
            echo_top_tracking._create_secondary_storm_id(
                PREV_SECONDARY_ID_NUMERIC)
        )

        self.assertTrue(this_id_string == SECONDARY_ID_STRING)
        self.assertTrue(this_id_numeric == PREV_SECONDARY_ID_NUMERIC + 1)

    def test_create_full_storm_id_same_day(self):
        """Ensures correct output from _create_full_storm_id.

        In this case the new storm occurs on the same SPC date as the previous
        storm to get a new ID.
        """

        this_id_string = echo_top_tracking._create_full_storm_id(
            primary_id_string=PRIMARY_ID_STRING_SAME_DAY,
            secondary_id_string=SECONDARY_ID_STRING)

        self.assertTrue(this_id_string == FULL_ID_STRING_SAME_DAY)

    def test_create_full_storm_id_next_day(self):
        """Ensures correct output from _create_full_storm_id.

        In this case the new storm *does not* occur on the same SPC date as the
        previous storm to get a new ID.
        """

        this_id_string = echo_top_tracking._create_full_storm_id(
            primary_id_string=PRIMARY_ID_STRING_NEXT_DAY,
            secondary_id_string=SECONDARY_ID_STRING)

        self.assertTrue(this_id_string == FULL_ID_STRING_NEXT_DAY)

    def test_local_maxima_to_tracks_mergers_1to2(self):
        """Ensures correct output from _local_maxima_to_tracks_mergers.

        In this case, linking maxima from the first and second times.
        """

        this_current_local_max_dict = copy.deepcopy(
            SECOND_LOCAL_MAX_DICT_UNLINKED)
        this_num_storm_objects = len(
            this_current_local_max_dict[echo_top_tracking.X_COORDS_KEY]
        )

        this_current_local_max_dict.update({
            echo_top_tracking.PRIMARY_IDS_KEY: [''] * this_num_storm_objects,
            echo_top_tracking.SECONDARY_IDS_KEY: [''] * this_num_storm_objects
        })

        this_dict = echo_top_tracking._local_maxima_to_tracks_mergers(
            current_local_max_dict=this_current_local_max_dict,
            previous_local_max_dict=FIRST_LOCAL_MAX_DICT_LINKED,
            current_to_previous_matrix=copy.deepcopy(
                CURRENT_TO_PREV_MATRIX_1TO2),
            prev_primary_id_numeric=PREV_PRIMARY_ID_1TO2_PREMERGE,
            prev_spc_date_string=PREV_SPC_DATE_STRING_1TO2_PREMERGE,
            prev_secondary_id_numeric=PREV_SECONDARY_ID_1TO2_PREMERGE)

        # TODO(thunderhoser): Verify the fucking mapping thing (old to new).

        this_current_local_max_dict = this_dict[
            echo_top_tracking.CURRENT_LOCAL_MAXIMA_KEY]
        this_current_to_prev_matrix = this_dict[
            echo_top_tracking.CURRENT_TO_PREV_MATRIX_KEY]
        prev_primary_id_numeric = this_dict[
            echo_top_tracking.PREVIOUS_PRIMARY_ID_KEY]
        prev_spc_date_string = this_dict[
            echo_top_tracking.PREVIOUS_SPC_DATE_KEY]
        prev_secondary_id_numeric = this_dict[
            echo_top_tracking.PREVIOUS_SECONDARY_ID_KEY]

        these_primary_id_strings = this_current_local_max_dict[
            echo_top_tracking.PRIMARY_IDS_KEY]
        these_secondary_id_strings = this_current_local_max_dict[
            echo_top_tracking.SECONDARY_IDS_KEY]

        self.assertTrue(
            these_primary_id_strings == SECOND_PRIMARY_ID_STRINGS_POSTMERGE
        )
        self.assertTrue(
            these_secondary_id_strings == SECOND_SECONDARY_ID_STRINGS_POSTMERGE
        )
        self.assertTrue(numpy.array_equal(
            this_current_to_prev_matrix, CURRENT_TO_PREV_MATRIX_1TO2_POSTMERGE
        ))
        self.assertTrue(
            prev_primary_id_numeric == PREV_PRIMARY_ID_1TO2_POSTMERGE
        )
        self.assertTrue(
            prev_spc_date_string == PREV_SPC_DATE_STRING_1TO2_POSTMERGE
        )
        self.assertTrue(
            prev_secondary_id_numeric == PREV_SECONDARY_ID_1TO2_POSTMERGE
        )

    def test_local_maxima_to_tracks_splits_1to2(self):
        """Ensures correct output from _local_maxima_to_tracks_splits.

        In this case, linking maxima from the first and second times.
        """

        this_current_local_max_dict = copy.deepcopy(
            SECOND_LOCAL_MAX_DICT_UNLINKED)

        this_current_local_max_dict.update({
            echo_top_tracking.PRIMARY_IDS_KEY:
                copy.deepcopy(SECOND_PRIMARY_ID_STRINGS_POSTMERGE),
            echo_top_tracking.SECONDARY_IDS_KEY:
                copy.deepcopy(SECOND_SECONDARY_ID_STRINGS_POSTMERGE)
        })

        this_dict = echo_top_tracking._local_maxima_to_tracks_splits(
            current_local_max_dict=this_current_local_max_dict,
            previous_local_max_dict=FIRST_LOCAL_MAX_DICT_LINKED,
            current_to_previous_matrix=copy.deepcopy(
                CURRENT_TO_PREV_MATRIX_1TO2_POSTMERGE),
            prev_secondary_id_numeric=PREV_SECONDARY_ID_1TO2_POSTMERGE)

        this_current_local_max_dict = this_dict[
            echo_top_tracking.CURRENT_LOCAL_MAXIMA_KEY]
        this_current_to_prev_matrix = this_dict[
            echo_top_tracking.CURRENT_TO_PREV_MATRIX_KEY]
        prev_secondary_id_numeric = this_dict[
            echo_top_tracking.PREVIOUS_SECONDARY_ID_KEY]

        these_primary_id_strings = this_current_local_max_dict[
            echo_top_tracking.PRIMARY_IDS_KEY]
        these_secondary_id_strings = this_current_local_max_dict[
            echo_top_tracking.SECONDARY_IDS_KEY]

        self.assertTrue(
            these_primary_id_strings == SECOND_PRIMARY_ID_STRINGS_POSTSPLIT
        )
        self.assertTrue(
            these_secondary_id_strings == SECOND_SECONDARY_ID_STRINGS_POSTSPLIT
        )
        self.assertTrue(numpy.array_equal(
            this_current_to_prev_matrix, CURRENT_TO_PREV_MATRIX_1TO2_POSTSPLIT
        ))
        self.assertTrue(
            prev_secondary_id_numeric == PREV_SECONDARY_ID_1TO2_POSTSPLIT
        )

    def test_local_maxima_to_tracks_simple_1to2(self):
        """Ensures correct output from _local_maxima_to_tracks_simple.

        In this case, linking maxima from the first and second times.
        """

        this_current_local_max_dict = copy.deepcopy(
            SECOND_LOCAL_MAX_DICT_UNLINKED)

        this_current_local_max_dict.update({
            echo_top_tracking.PRIMARY_IDS_KEY:
                copy.deepcopy(SECOND_PRIMARY_ID_STRINGS_POSTSPLIT),
            echo_top_tracking.SECONDARY_IDS_KEY:
                copy.deepcopy(SECOND_SECONDARY_ID_STRINGS_POSTSPLIT)
        })

        this_dict = echo_top_tracking._local_maxima_to_tracks_simple(
            current_local_max_dict=this_current_local_max_dict,
            previous_local_max_dict=FIRST_LOCAL_MAX_DICT_LINKED,
            current_to_previous_matrix=copy.deepcopy(
                CURRENT_TO_PREV_MATRIX_1TO2_POSTSPLIT),
            prev_primary_id_numeric=PREV_PRIMARY_ID_1TO2_POSTSPLIT,
            prev_spc_date_string=PREV_SPC_DATE_STRING_1TO2_POSTSPLIT,
            prev_secondary_id_numeric=PREV_SECONDARY_ID_1TO2_POSTSPLIT)

        this_current_local_max_dict = this_dict[
            echo_top_tracking.CURRENT_LOCAL_MAXIMA_KEY]
        prev_primary_id_numeric = this_dict[
            echo_top_tracking.PREVIOUS_PRIMARY_ID_KEY]
        prev_spc_date_string = this_dict[
            echo_top_tracking.PREVIOUS_SPC_DATE_KEY]
        prev_secondary_id_numeric = this_dict[
            echo_top_tracking.PREVIOUS_SECONDARY_ID_KEY]

        these_primary_id_strings = this_current_local_max_dict[
            echo_top_tracking.PRIMARY_IDS_KEY]
        these_secondary_id_strings = this_current_local_max_dict[
            echo_top_tracking.SECONDARY_IDS_KEY]

        self.assertTrue(these_primary_id_strings == SECOND_PRIMARY_ID_STRINGS)
        self.assertTrue(
            these_secondary_id_strings == SECOND_SECONDARY_ID_STRINGS
        )
        self.assertTrue(
            prev_primary_id_numeric == PREV_PRIMARY_ID_1TO2_POST
        )
        self.assertTrue(
            prev_spc_date_string == PREV_SPC_DATE_STRING_1TO2_POST
        )
        self.assertTrue(
            prev_secondary_id_numeric == PREV_SECONDARY_ID_1TO2_POST
        )

    def test_local_maxima_to_storm_tracks(self):
        """Ensures correct output from _local_maxima_to_storm_tracks."""

        this_table = echo_top_tracking._local_maxima_to_storm_tracks(
            [FIRST_LOCAL_MAX_DICT_LINKED, SECOND_LOCAL_MAX_DICT_LINKED]
        )

        print this_table


if __name__ == '__main__':
    unittest.main()
