"""Unit tests for track_reanalysis.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import track_reanalysis
from gewittergefahr.gg_utils import temporal_tracking
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

TOLERANCE = 1e-6

ORIG_X_COORDS_METRES = numpy.array([
    20, 24, 28, 32,
    36, 40, 44,
    36, 40, 44, 48,
    10, 15, 20, 25, 30, 35,
    8, 13, 18, 23, 28, 33,
    40, 41,
    20, 22, 24, 26, 28, 30, 32,
    0, 5, 10, 15, 20,
    25, 28, 31, 34, 37, 40
], dtype=float)

ORIG_Y_COORDS_METRES = numpy.array([
    100, 101, 102, 103,
    105, 110, 115,
    103, 101, 99, 97,
    70, 70, 70, 70, 70, 70,
    55, 57, 59, 61, 63, 65,
    67, 68,
    30, 32, 34, 36, 38, 40, 42,
    0, -7.5, -15, -22.5, -30,
    -30, -31.5, -33, -34.5, -36, -37.5
])

ORIG_X_VELOCITIES_M_S01 = numpy.array([
    4, 4, 4, 4,
    4, 4, 4,
    4, 4, 4, 4,
    5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5,
    1, 1,
    2, 2, 2, 2, 2, 2, 2,
    5, 5, 5, 5, 5,
    3, 3, 3, 3, 3, 3
], dtype=float)

ORIG_Y_VELOCITIES_M_S01 = numpy.array([
    1, 1, 1, 1,
    5, 5, 5,
    -2, -2, -2, -2,
    0, 0, 0, 0, 0, 0,
    2, 2, 2, 2, 2, 2,
    1, 1,
    2, 2, 2, 2, 2, 2, 2,
    -7.5, -7.5, -7.5, -7.5, -7.5,
    -1.5, -1.5, -1.5, -1.5, -1.5, -1.5
])

ORIG_TIMES_UNIX_SEC = numpy.array([
    0, 1, 2, 3,
    5, 6, 7,
    5, 6, 7, 8,
    2, 3, 4, 5, 6, 7,
    2, 3, 4, 5, 6, 7,
    9, 10,
    3, 4, 5, 6, 7, 8, 9,
    5, 6, 7, 8, 9,
    12, 13, 14, 15, 16, 17
], dtype=int)

ORIG_PRIMARY_ID_STRINGS = [
    'A', 'A', 'A', 'A',
    'B', 'B', 'B',
    'C', 'C', 'C', 'C',
    'D', 'D', 'D', 'D', 'D', 'D',
    'E', 'E', 'E', 'E', 'E', 'E',
    'F', 'F',
    'G', 'G', 'G', 'G', 'G', 'G', 'G',
    'H', 'H', 'H', 'H', 'H',
    'J', 'J', 'J', 'J', 'J', 'J'
]

ORIG_SECONDARY_ID_STRINGS = [
    '01', '01', '01', '01',
    '02', '02', '02',
    '03', '03', '03', '03',
    '04', '04', '04', '04', '04', '04',
    '05', '05', '05', '05', '05', '05',
    '06', '06',
    '07', '07', '07', '07', '07', '07', '07',
    '08', '08', '08', '08', '08',
    '10', '10', '10', '10', '10', '10'
]

ORIG_FIRST_PREV_SECONDARY_IDS = [
    '', '01', '01', '01',
    '', '02', '02',
    '', '03', '03', '03',
    '', '04', '04', '04', '04', '04',
    '', '05', '05', '05', '05', '05',
    '', '06',
    '', '07', '07', '07', '07', '07', '07',
    '', '08', '08', '08', '08',
    '', '10', '10', '10', '10', '10'
]

ORIG_SECOND_PREV_SECONDARY_IDS = [''] * len(ORIG_TIMES_UNIX_SEC)

ORIG_FIRST_NEXT_SECONDARY_IDS = [
    '01', '01', '01', '',
    '02', '02', '',
    '03', '03', '03', '',
    '04', '04', '04', '04', '04', '',
    '05', '05', '05', '05', '05', '',
    '06', '',
    '07', '07', '07', '07', '07', '07', '',
    '08', '08', '08', '08', '',
    '10', '10', '10', '10', '10', ''
]

ORIG_SECOND_NEXT_SECONDARY_IDS = [''] * len(ORIG_TIMES_UNIX_SEC)

THIS_DICT = {
    temporal_tracking.CENTROID_X_COLUMN: ORIG_X_COORDS_METRES,
    temporal_tracking.CENTROID_Y_COLUMN: ORIG_Y_COORDS_METRES,
    temporal_tracking.X_VELOCITY_COLUMN: ORIG_X_VELOCITIES_M_S01,
    temporal_tracking.Y_VELOCITY_COLUMN: ORIG_Y_VELOCITIES_M_S01,
    tracking_utils.VALID_TIME_COLUMN: ORIG_TIMES_UNIX_SEC,
    tracking_utils.PRIMARY_ID_COLUMN: ORIG_PRIMARY_ID_STRINGS,
    tracking_utils.SECONDARY_ID_COLUMN: ORIG_SECONDARY_ID_STRINGS,
    tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN:
        ORIG_FIRST_PREV_SECONDARY_IDS,
    tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN:
        ORIG_SECOND_PREV_SECONDARY_IDS,
    tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN:
        ORIG_FIRST_NEXT_SECONDARY_IDS,
    tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN:
        ORIG_SECOND_NEXT_SECONDARY_IDS
}

ORIG_STORM_OBJECT_TABLE = track_reanalysis._update_full_ids(
    pandas.DataFrame.from_dict(THIS_DICT)
)

COLUMNS_TO_COMPARE = list(ORIG_STORM_OBJECT_TABLE)
COLUMNS_TO_COMPARE.remove(temporal_tracking.X_VELOCITY_COLUMN)
COLUMNS_TO_COMPARE.remove(temporal_tracking.Y_VELOCITY_COLUMN)

# The following constants are used to test _handle_collinear_splits.
EARLY_ROWS_FOR_SPLIT = numpy.array([20, 4, 3, 1], dtype=int)
LATE_ROWS_FOR_SPLIT = numpy.array([30, 4, 7, 0, 10], dtype=int)

LATE_TO_EARLY_MATRIX_PRESPLIT = numpy.array([
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
], dtype=bool)

THESE_ID_STRINGS, THESE_FIRST_ROWS = numpy.unique(
    ORIG_STORM_OBJECT_TABLE[tracking_utils.PRIMARY_ID_COLUMN].values,
    return_index=True)

ID_TO_FIRST_ROW_DICT_PRESPLIT = dict(zip(THESE_ID_STRINGS, THESE_FIRST_ROWS))

THESE_ID_STRINGS, THESE_LAST_ROWS = numpy.unique(
    ORIG_STORM_OBJECT_TABLE[tracking_utils.PRIMARY_ID_COLUMN].values[::-1],
    return_index=True)

THIS_NUM_ROWS = len(ORIG_STORM_OBJECT_TABLE.index)
THESE_LAST_ROWS = THIS_NUM_ROWS - THESE_LAST_ROWS - 1
ID_TO_LAST_ROW_DICT_PRESPLIT = dict(zip(THESE_ID_STRINGS, THESE_LAST_ROWS))

LATE_TO_EARLY_MATRIX_POSTSPLIT = numpy.full((5, 4), False, dtype=bool)

ID_TO_FIRST_ROW_DICT_POSTSPLIT = copy.deepcopy(ID_TO_FIRST_ROW_DICT_PRESPLIT)
ID_TO_FIRST_ROW_DICT_POSTSPLIT['B'] = -1
ID_TO_FIRST_ROW_DICT_POSTSPLIT['C'] = -1

ID_TO_LAST_ROW_DICT_POSTSPLIT = copy.deepcopy(ID_TO_LAST_ROW_DICT_PRESPLIT)
ID_TO_LAST_ROW_DICT_POSTSPLIT['A'] = 10
ID_TO_LAST_ROW_DICT_POSTSPLIT['B'] = -1
ID_TO_LAST_ROW_DICT_POSTSPLIT['C'] = -1

THESE_PRIMARY_ID_STRINGS = [
    'A', 'A', 'A', 'A',
    'A', 'A', 'A',
    'A', 'A', 'A', 'A',
    'D', 'D', 'D', 'D', 'D', 'D',
    'E', 'E', 'E', 'E', 'E', 'E',
    'F', 'F',
    'G', 'G', 'G', 'G', 'G', 'G', 'G',
    'H', 'H', 'H', 'H', 'H',
    'J', 'J', 'J', 'J', 'J', 'J'
]

THESE_FIRST_PREV_SECONDARY_IDS = copy.deepcopy(ORIG_FIRST_PREV_SECONDARY_IDS)
THESE_FIRST_PREV_SECONDARY_IDS[4] = '01'
THESE_FIRST_PREV_SECONDARY_IDS[7] = '01'

THESE_FIRST_NEXT_SECONDARY_IDS = copy.deepcopy(ORIG_FIRST_NEXT_SECONDARY_IDS)
THESE_FIRST_NEXT_SECONDARY_IDS[3] = '02'

THESE_SECOND_NEXT_SECONDARY_IDS = copy.deepcopy(ORIG_SECOND_NEXT_SECONDARY_IDS)
THESE_SECOND_NEXT_SECONDARY_IDS[3] = '03'

STORM_OBJECT_TABLE_POSTSPLIT = copy.deepcopy(ORIG_STORM_OBJECT_TABLE)

STORM_OBJECT_TABLE_POSTSPLIT = STORM_OBJECT_TABLE_POSTSPLIT.assign(**{
    tracking_utils.PRIMARY_ID_COLUMN: THESE_PRIMARY_ID_STRINGS,
    tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN:
        THESE_FIRST_PREV_SECONDARY_IDS,
    tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN:
        THESE_FIRST_NEXT_SECONDARY_IDS,
    tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN:
        THESE_SECOND_NEXT_SECONDARY_IDS
})

# The following constants are used to test _handle_collinear_mergers.
EARLY_ROWS_FOR_MERGER = numpy.array([30, 16, 22, 0, 1], dtype=int)
LATE_ROWS_FOR_MERGER = numpy.array([5, 23, 6, 28], dtype=int)

LATE_TO_EARLY_MATRIX_PREMERGE = numpy.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
], dtype=bool)

LATE_TO_EARLY_MATRIX_POSTMERGE = numpy.full((4, 5), False, dtype=bool)
STORM_OBJECT_TABLE_PREMERGE = copy.deepcopy(STORM_OBJECT_TABLE_POSTSPLIT)
ID_TO_FIRST_ROW_DICT_PREMERGE = copy.deepcopy(ID_TO_FIRST_ROW_DICT_POSTSPLIT)
ID_TO_LAST_ROW_DICT_PREMERGE = copy.deepcopy(ID_TO_LAST_ROW_DICT_POSTSPLIT)

ID_TO_FIRST_ROW_DICT_POSTMERGE = copy.deepcopy(ID_TO_FIRST_ROW_DICT_PREMERGE)
ID_TO_FIRST_ROW_DICT_POSTMERGE['D'] = -1
ID_TO_FIRST_ROW_DICT_POSTMERGE['E'] = -1
ID_TO_FIRST_ROW_DICT_POSTMERGE['F'] = 11

ID_TO_LAST_ROW_DICT_POSTMERGE = copy.deepcopy(ID_TO_LAST_ROW_DICT_PREMERGE)
ID_TO_LAST_ROW_DICT_POSTMERGE['D'] = -1
ID_TO_LAST_ROW_DICT_POSTMERGE['E'] = -1

THESE_PRIMARY_ID_STRINGS = [
    'A', 'A', 'A', 'A',
    'A', 'A', 'A',
    'A', 'A', 'A', 'A',
    'F', 'F', 'F', 'F', 'F', 'F',
    'F', 'F', 'F', 'F', 'F', 'F',
    'F', 'F',
    'G', 'G', 'G', 'G', 'G', 'G', 'G',
    'H', 'H', 'H', 'H', 'H',
    'J', 'J', 'J', 'J', 'J', 'J'
]

THESE_FIRST_PREV_SECONDARY_IDS = copy.deepcopy(
    STORM_OBJECT_TABLE_POSTSPLIT[
        tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN].values
)
THESE_FIRST_PREV_SECONDARY_IDS[23] = '04'

THESE_SECOND_PREV_SECONDARY_IDS = copy.deepcopy(
    STORM_OBJECT_TABLE_POSTSPLIT[
        tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN].values
)
THESE_SECOND_PREV_SECONDARY_IDS[23] = '05'

THESE_FIRST_NEXT_SECONDARY_IDS = copy.deepcopy(
    STORM_OBJECT_TABLE_POSTSPLIT[
        tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN].values
)
THESE_FIRST_NEXT_SECONDARY_IDS[16] = '06'
THESE_FIRST_NEXT_SECONDARY_IDS[22] = '06'

STORM_OBJECT_TABLE_POSTMERGE = copy.deepcopy(STORM_OBJECT_TABLE_PREMERGE)

STORM_OBJECT_TABLE_POSTMERGE = STORM_OBJECT_TABLE_POSTMERGE.assign(**{
    tracking_utils.PRIMARY_ID_COLUMN: THESE_PRIMARY_ID_STRINGS,
    tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN:
        THESE_FIRST_PREV_SECONDARY_IDS,
    tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN:
        THESE_SECOND_PREV_SECONDARY_IDS,
    tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN:
        THESE_FIRST_NEXT_SECONDARY_IDS
})

# The following constants are used to test _handle_collinear_1to1_joins.
EARLY_ROWS_FOR_SIMPLE = numpy.array([36], dtype=int)
LATE_ROWS_FOR_SIMPLE = numpy.array([37], dtype=int)
LATE_TO_EARLY_MATRIX_PRESIMPLE = numpy.full((1, 1), True, dtype=bool)

STORM_OBJECT_TABLE_PRESIMPLE = copy.deepcopy(STORM_OBJECT_TABLE_POSTMERGE)
ID_TO_FIRST_ROW_DICT_PRESIMPLE = copy.deepcopy(ID_TO_FIRST_ROW_DICT_POSTMERGE)
ID_TO_LAST_ROW_DICT_PRESIMPLE = copy.deepcopy(ID_TO_LAST_ROW_DICT_POSTMERGE)

ID_TO_FIRST_ROW_DICT_POSTSIMPLE = copy.deepcopy(ID_TO_FIRST_ROW_DICT_PRESIMPLE)
ID_TO_FIRST_ROW_DICT_POSTSIMPLE['H'] = -1
ID_TO_FIRST_ROW_DICT_POSTSIMPLE['J'] = 32

ID_TO_LAST_ROW_DICT_POSTSIMPLE = copy.deepcopy(ID_TO_LAST_ROW_DICT_PRESIMPLE)
ID_TO_LAST_ROW_DICT_POSTSIMPLE['H'] = -1

THESE_PRIMARY_ID_STRINGS = [
    'A', 'A', 'A', 'A',
    'A', 'A', 'A',
    'A', 'A', 'A', 'A',
    'F', 'F', 'F', 'F', 'F', 'F',
    'F', 'F', 'F', 'F', 'F', 'F',
    'F', 'F',
    'G', 'G', 'G', 'G', 'G', 'G', 'G',
    'J', 'J', 'J', 'J', 'J',
    'J', 'J', 'J', 'J', 'J', 'J'
]

THESE_SECONDARY_ID_STRINGS = [
    '01', '01', '01', '01',
    '02', '02', '02',
    '03', '03', '03', '03',
    '04', '04', '04', '04', '04', '04',
    '05', '05', '05', '05', '05', '05',
    '06', '06',
    '07', '07', '07', '07', '07', '07', '07',
    '10', '10', '10', '10', '10',
    '10', '10', '10', '10', '10', '10'
]

THESE_FIRST_PREV_SECONDARY_IDS = copy.deepcopy(
    STORM_OBJECT_TABLE_POSTMERGE[
        tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN].values
)
THESE_FIRST_PREV_SECONDARY_IDS[37] = '10'
THESE_FIRST_PREV_SECONDARY_IDS[36] = '10'
THESE_FIRST_PREV_SECONDARY_IDS[35] = '10'
THESE_FIRST_PREV_SECONDARY_IDS[34] = '10'
THESE_FIRST_PREV_SECONDARY_IDS[33] = '10'

THESE_FIRST_NEXT_SECONDARY_IDS = copy.deepcopy(
    STORM_OBJECT_TABLE_POSTMERGE[
        tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN].values
)
THESE_FIRST_NEXT_SECONDARY_IDS[32] = '10'
THESE_FIRST_NEXT_SECONDARY_IDS[33] = '10'
THESE_FIRST_NEXT_SECONDARY_IDS[34] = '10'
THESE_FIRST_NEXT_SECONDARY_IDS[35] = '10'
THESE_FIRST_NEXT_SECONDARY_IDS[36] = '10'

STORM_OBJECT_TABLE_POSTSIMPLE = copy.deepcopy(STORM_OBJECT_TABLE_PRESIMPLE)

STORM_OBJECT_TABLE_POSTSIMPLE = STORM_OBJECT_TABLE_POSTSIMPLE.assign(**{
    tracking_utils.PRIMARY_ID_COLUMN: THESE_PRIMARY_ID_STRINGS,
    tracking_utils.SECONDARY_ID_COLUMN: THESE_SECONDARY_ID_STRINGS,
    tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN:
        THESE_FIRST_PREV_SECONDARY_IDS,
    tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN:
        THESE_FIRST_NEXT_SECONDARY_IDS
})

# The following constants are used to test join_collinear_tracks.
STORM_OBJECT_TABLE_1SEC = copy.deepcopy(ORIG_STORM_OBJECT_TABLE)
STORM_OBJECT_TABLE_2SEC_5METRES = copy.deepcopy(STORM_OBJECT_TABLE_POSTMERGE)
STORM_OBJECT_TABLE_2SEC_5METRES = track_reanalysis._update_full_ids(
    STORM_OBJECT_TABLE_2SEC_5METRES)

STORM_OBJECT_TABLE_3SEC_5METRES = copy.deepcopy(STORM_OBJECT_TABLE_2SEC_5METRES)
STORM_OBJECT_TABLE_2SEC_10METRES = copy.deepcopy(
    STORM_OBJECT_TABLE_2SEC_5METRES)
STORM_OBJECT_TABLE_3SEC_10METRES = copy.deepcopy(STORM_OBJECT_TABLE_POSTSIMPLE)
STORM_OBJECT_TABLE_3SEC_10METRES = track_reanalysis._update_full_ids(
    STORM_OBJECT_TABLE_3SEC_10METRES)

STORM_OBJECT_TABLE_2SEC_30METRES = copy.deepcopy(
    STORM_OBJECT_TABLE_2SEC_5METRES)
STORM_OBJECT_TABLE_3SEC_30METRES = copy.deepcopy(
    STORM_OBJECT_TABLE_2SEC_5METRES)

THESE_PRIMARY_ID_STRINGS = [
    'A', 'A', 'A', 'A',
    'A', 'A', 'A',
    'A', 'A', 'A', 'A',
    'F', 'F', 'F', 'F', 'F', 'F',
    'F', 'F', 'F', 'F', 'F', 'F',
    'F', 'F',
    'J', 'J', 'J', 'J', 'J', 'J', 'J',
    'J', 'J', 'J', 'J', 'J',
    'J', 'J', 'J', 'J', 'J', 'J'
]

THESE_SECONDARY_ID_STRINGS = [
    '01', '01', '01', '01',
    '02', '02', '02',
    '03', '03', '03', '03',
    '04', '04', '04', '04', '04', '04',
    '05', '05', '05', '05', '05', '05',
    '06', '06',
    '07', '07', '07', '07', '07', '07', '07',
    '08', '08', '08', '08', '08',
    '10', '10', '10', '10', '10', '10'
]

THESE_FIRST_PREV_SECONDARY_IDS = copy.deepcopy(
    STORM_OBJECT_TABLE_POSTMERGE[
        tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN].values
)
THESE_FIRST_PREV_SECONDARY_IDS[37] = '07'

THESE_SECOND_PREV_SECONDARY_IDS = copy.deepcopy(
    STORM_OBJECT_TABLE_POSTMERGE[
        tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN].values
)
THESE_SECOND_PREV_SECONDARY_IDS[37] = '08'

THESE_FIRST_NEXT_SECONDARY_IDS = copy.deepcopy(
    STORM_OBJECT_TABLE_POSTMERGE[
        tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN].values
)
THESE_FIRST_NEXT_SECONDARY_IDS[31] = '10'
THESE_FIRST_NEXT_SECONDARY_IDS[36] = '10'

STORM_OBJECT_TABLE_3SEC_30METRES = STORM_OBJECT_TABLE_3SEC_30METRES.assign(**{
    tracking_utils.PRIMARY_ID_COLUMN: THESE_PRIMARY_ID_STRINGS,
    tracking_utils.SECONDARY_ID_COLUMN: THESE_SECONDARY_ID_STRINGS,
    tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN:
        THESE_FIRST_PREV_SECONDARY_IDS,
    tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN:
        THESE_SECOND_PREV_SECONDARY_IDS,
    tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN:
        THESE_FIRST_NEXT_SECONDARY_IDS
})

STORM_OBJECT_TABLE_3SEC_30METRES = track_reanalysis._update_full_ids(
    STORM_OBJECT_TABLE_3SEC_30METRES)


class TrackReanalysisTests(unittest.TestCase):
    """Each method is a unit test for track_reanalysis.py."""

    def test_handle_collinear_splits(self):
        """Ensures correct output from _handle_collinear_splits."""

        this_dict = track_reanalysis._handle_collinear_splits(
            storm_object_table=copy.deepcopy(ORIG_STORM_OBJECT_TABLE),
            early_rows=EARLY_ROWS_FOR_SPLIT, late_rows=LATE_ROWS_FOR_SPLIT,
            late_to_early_matrix=copy.deepcopy(LATE_TO_EARLY_MATRIX_PRESPLIT),
            primary_id_to_first_row_dict=copy.deepcopy(
                ID_TO_FIRST_ROW_DICT_PRESPLIT),
            primary_id_to_last_row_dict=copy.deepcopy(
                ID_TO_LAST_ROW_DICT_PRESPLIT)
        )

        this_storm_object_table = this_dict[
            track_reanalysis.STORM_OBJECT_TABLE_KEY]
        this_late_to_early_matrix = this_dict[
            track_reanalysis.LATE_TO_EARLY_KEY]
        this_id_to_first_row_dict = this_dict[
            track_reanalysis.ID_TO_FIRST_ROW_KEY]
        this_id_to_last_row_dict = this_dict[
            track_reanalysis.ID_TO_LAST_ROW_KEY]

        self.assertTrue(numpy.array_equal(
            this_late_to_early_matrix, LATE_TO_EARLY_MATRIX_POSTSPLIT
        ))
        self.assertTrue(
            this_id_to_first_row_dict == ID_TO_FIRST_ROW_DICT_POSTSPLIT
        )
        self.assertTrue(
            this_id_to_last_row_dict == ID_TO_LAST_ROW_DICT_POSTSPLIT
        )
        self.assertTrue(
            this_storm_object_table.equals(STORM_OBJECT_TABLE_POSTSPLIT)
        )

    def test_handle_collinear_mergers(self):
        """Ensures correct output from _handle_collinear_mergers."""

        this_dict = track_reanalysis._handle_collinear_mergers(
            storm_object_table=copy.deepcopy(STORM_OBJECT_TABLE_PREMERGE),
            early_rows=EARLY_ROWS_FOR_MERGER, late_rows=LATE_ROWS_FOR_MERGER,
            late_to_early_matrix=copy.deepcopy(LATE_TO_EARLY_MATRIX_PREMERGE),
            primary_id_to_first_row_dict=copy.deepcopy(
                ID_TO_FIRST_ROW_DICT_PREMERGE),
            primary_id_to_last_row_dict=copy.deepcopy(
                ID_TO_LAST_ROW_DICT_PREMERGE)
        )

        this_storm_object_table = this_dict[
            track_reanalysis.STORM_OBJECT_TABLE_KEY]
        this_late_to_early_matrix = this_dict[
            track_reanalysis.LATE_TO_EARLY_KEY]
        this_id_to_first_row_dict = this_dict[
            track_reanalysis.ID_TO_FIRST_ROW_KEY]
        this_id_to_last_row_dict = this_dict[
            track_reanalysis.ID_TO_LAST_ROW_KEY]

        self.assertTrue(numpy.array_equal(
            this_late_to_early_matrix, LATE_TO_EARLY_MATRIX_POSTMERGE
        ))
        self.assertTrue(
            this_id_to_first_row_dict == ID_TO_FIRST_ROW_DICT_POSTMERGE
        )
        self.assertTrue(
            this_id_to_last_row_dict == ID_TO_LAST_ROW_DICT_POSTMERGE
        )
        self.assertTrue(
            this_storm_object_table.equals(STORM_OBJECT_TABLE_POSTMERGE)
        )

    def test_handle_collinear_1to1_joins(self):
        """Ensures correct output from _handle_collinear_1to1_joins."""

        this_dict = track_reanalysis._handle_collinear_1to1_joins(
            storm_object_table=copy.deepcopy(STORM_OBJECT_TABLE_PRESIMPLE),
            early_rows=EARLY_ROWS_FOR_SIMPLE, late_rows=LATE_ROWS_FOR_SIMPLE,
            late_to_early_matrix=copy.deepcopy(LATE_TO_EARLY_MATRIX_PRESIMPLE),
            primary_id_to_first_row_dict=copy.deepcopy(
                ID_TO_FIRST_ROW_DICT_PRESIMPLE),
            primary_id_to_last_row_dict=copy.deepcopy(
                ID_TO_LAST_ROW_DICT_PRESIMPLE)
        )

        this_storm_object_table = this_dict[
            track_reanalysis.STORM_OBJECT_TABLE_KEY]
        this_id_to_first_row_dict = this_dict[
            track_reanalysis.ID_TO_FIRST_ROW_KEY]
        this_id_to_last_row_dict = this_dict[
            track_reanalysis.ID_TO_LAST_ROW_KEY]

        self.assertTrue(
            this_id_to_first_row_dict == ID_TO_FIRST_ROW_DICT_POSTSIMPLE
        )
        self.assertTrue(
            this_id_to_last_row_dict == ID_TO_LAST_ROW_DICT_POSTSIMPLE
        )
        self.assertTrue(
            this_storm_object_table.equals(STORM_OBJECT_TABLE_POSTSIMPLE)
        )

    def test_join_collinear_tracks_1sec_5metres(self):
        """Ensures correct output from join_collinear_tracks.

        In this case, max join time = 1 second and max join error = 5 m/s.
        """

        this_storm_object_table = track_reanalysis.join_collinear_tracks(
            storm_object_table=copy.deepcopy(ORIG_STORM_OBJECT_TABLE),
            first_late_time_unix_sec=numpy.min(ORIG_TIMES_UNIX_SEC),
            last_late_time_unix_sec=numpy.max(ORIG_TIMES_UNIX_SEC),
            max_join_time_seconds=1, max_join_error_m_s01=5.)

        self.assertTrue(
            this_storm_object_table[COLUMNS_TO_COMPARE].equals(
                STORM_OBJECT_TABLE_1SEC[COLUMNS_TO_COMPARE]
            )
        )

    def test_join_collinear_tracks_2sec_5metres(self):
        """Ensures correct output from join_collinear_tracks.

        In this case, max join time = 2 seconds and max join error = 5 m/s.
        """

        this_storm_object_table = track_reanalysis.join_collinear_tracks(
            storm_object_table=copy.deepcopy(ORIG_STORM_OBJECT_TABLE),
            first_late_time_unix_sec=numpy.min(ORIG_TIMES_UNIX_SEC),
            last_late_time_unix_sec=numpy.max(ORIG_TIMES_UNIX_SEC),
            max_join_time_seconds=2, max_join_error_m_s01=5.)

        self.assertTrue(
            this_storm_object_table[COLUMNS_TO_COMPARE].equals(
                STORM_OBJECT_TABLE_2SEC_5METRES[COLUMNS_TO_COMPARE]
            )
        )

    def test_join_collinear_tracks_3sec_5metres(self):
        """Ensures correct output from join_collinear_tracks.

        In this case, max join time = 3 seconds and max join error = 5 m/s.
        """

        this_storm_object_table = track_reanalysis.join_collinear_tracks(
            storm_object_table=copy.deepcopy(ORIG_STORM_OBJECT_TABLE),
            first_late_time_unix_sec=numpy.min(ORIG_TIMES_UNIX_SEC),
            last_late_time_unix_sec=numpy.max(ORIG_TIMES_UNIX_SEC),
            max_join_time_seconds=3, max_join_error_m_s01=5.)

        self.assertTrue(
            this_storm_object_table[COLUMNS_TO_COMPARE].equals(
                STORM_OBJECT_TABLE_3SEC_5METRES[COLUMNS_TO_COMPARE]
            )
        )

    def test_join_collinear_tracks_2sec_10metres(self):
        """Ensures correct output from join_collinear_tracks.

        In this case, max join time = 2 seconds and max join error = 10 m/s.
        """

        this_storm_object_table = track_reanalysis.join_collinear_tracks(
            storm_object_table=copy.deepcopy(ORIG_STORM_OBJECT_TABLE),
            first_late_time_unix_sec=numpy.min(ORIG_TIMES_UNIX_SEC),
            last_late_time_unix_sec=numpy.max(ORIG_TIMES_UNIX_SEC),
            max_join_time_seconds=2, max_join_error_m_s01=10.)

        self.assertTrue(
            this_storm_object_table[COLUMNS_TO_COMPARE].equals(
                STORM_OBJECT_TABLE_2SEC_10METRES[COLUMNS_TO_COMPARE]
            )
        )

    def test_join_collinear_tracks_3sec_10metres(self):
        """Ensures correct output from join_collinear_tracks.

        In this case, max join time = 3 seconds and max join error = 10 m/s.
        """

        this_storm_object_table = track_reanalysis.join_collinear_tracks(
            storm_object_table=copy.deepcopy(ORIG_STORM_OBJECT_TABLE),
            first_late_time_unix_sec=numpy.min(ORIG_TIMES_UNIX_SEC),
            last_late_time_unix_sec=numpy.max(ORIG_TIMES_UNIX_SEC),
            max_join_time_seconds=3, max_join_error_m_s01=10.)

        self.assertTrue(
            this_storm_object_table[COLUMNS_TO_COMPARE].equals(
                STORM_OBJECT_TABLE_3SEC_10METRES[COLUMNS_TO_COMPARE]
            )
        )

    def test_join_collinear_tracks_2sec_30metres(self):
        """Ensures correct output from join_collinear_tracks.

        In this case, max join time = 2 seconds and max join error = 30 m/s.
        """

        this_storm_object_table = track_reanalysis.join_collinear_tracks(
            storm_object_table=copy.deepcopy(ORIG_STORM_OBJECT_TABLE),
            first_late_time_unix_sec=numpy.min(ORIG_TIMES_UNIX_SEC),
            last_late_time_unix_sec=numpy.max(ORIG_TIMES_UNIX_SEC),
            max_join_time_seconds=2, max_join_error_m_s01=30.)

        self.assertTrue(
            this_storm_object_table[COLUMNS_TO_COMPARE].equals(
                STORM_OBJECT_TABLE_2SEC_30METRES[COLUMNS_TO_COMPARE]
            )
        )

    def test_join_collinear_tracks_3sec_30metres(self):
        """Ensures correct output from join_collinear_tracks.

        In this case, max join time = 3 seconds and max join error = 30 m/s.
        """

        this_storm_object_table = track_reanalysis.join_collinear_tracks(
            storm_object_table=copy.deepcopy(ORIG_STORM_OBJECT_TABLE),
            first_late_time_unix_sec=numpy.min(ORIG_TIMES_UNIX_SEC),
            last_late_time_unix_sec=numpy.max(ORIG_TIMES_UNIX_SEC),
            max_join_time_seconds=3, max_join_error_m_s01=30.)

        self.assertTrue(
            this_storm_object_table[COLUMNS_TO_COMPARE].equals(
                STORM_OBJECT_TABLE_3SEC_30METRES[COLUMNS_TO_COMPARE]
            )
        )


if __name__ == '__main__':
    unittest.main()
