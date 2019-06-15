"""Unit tests for tornado_io.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_io import tornado_io
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion

TOLERANCE = 1e-6

# The following constants are used to test _is_valid_fujita_rating.
F_SCALE_RATING_ALL_CAPS = 'F5'
F_SCALE_RATING_NO_CAPS = 'f4'
F_SCALE_RATING_LEADING_ZERO = 'F02'
F_SCALE_RATING_TOO_MANY_LETTERS = 'fF3'
F_SCALE_RATING_TOO_LOW = 'F-1'
F_SCALE_RATING_TOO_HIGH = 'F6'

EF_SCALE_RATING_ALL_CAPS = 'EF5'
EF_SCALE_RATING_NO_CAPS = 'ef4'
EF_SCALE_RATING_SOME_CAPS = 'eF3'
EF_SCALE_RATING_LEADING_ZERO = 'EF01'
EF_SCALE_RATING_TOO_MANY_LETTERS = 'EFF2'
EF_SCALE_RATING_TOO_LOW = 'EF-1'
EF_SCALE_RATING_TOO_HIGH = 'EF6'

# The following constants are used to test create_tornado_id.
TORNADO_START_TIME_UNIX_SEC = 1559857972  # 215252 UTC 6 Jun 2019
TORNADO_START_LATITUDE_DEG = 53.526196
TORNADO_START_LONGITUDE_DEG = 246.479111

TORNADO_ID_STRING = '2019-06-06-215252_53.526N_246.479E'

# The following constants are used to test remove_invalid_reports.
LATITUDES_DEG = numpy.array([-800., 20., 25., 30., 35., 40., 45.])
LONGITUDES_DEG = numpy.array([-100., -210., 270., -85., 280., -75., 290.])
UNIX_TIMES_SEC = numpy.array([1, 2, -666, 4, 5, 6, 7])
FUJITA_SCALE_RATINGS = ['EF0', 'EF1', 'F2', 'EF7', 'EF3', 'EF4', 'EF5']
TORNADO_WIDTHS_METRES = numpy.array(
    [1., 10., 100., 1000., 9999., 200., numpy.nan])

THIS_DICT = {
    tornado_io.START_LAT_COLUMN: LATITUDES_DEG,
    tornado_io.END_LAT_COLUMN: LATITUDES_DEG,
    tornado_io.START_LNG_COLUMN: LONGITUDES_DEG,
    tornado_io.END_LNG_COLUMN: LONGITUDES_DEG,
    tornado_io.START_TIME_COLUMN: UNIX_TIMES_SEC,
    tornado_io.END_TIME_COLUMN: UNIX_TIMES_SEC,
    tornado_io.FUJITA_RATING_COLUMN: FUJITA_SCALE_RATINGS,
    tornado_io.WIDTH_COLUMN: TORNADO_WIDTHS_METRES
}
TORNADO_TABLE_WITH_INVALID_ROWS = pandas.DataFrame.from_dict(THIS_DICT)

GOOD_ROWS = numpy.array([4, 5], dtype=int)
TORNADO_TABLE_NO_INVALID_ROWS = TORNADO_TABLE_WITH_INVALID_ROWS.iloc[GOOD_ROWS]

TORNADO_TABLE_NO_INVALID_ROWS = TORNADO_TABLE_NO_INVALID_ROWS.assign(**{
    tornado_io.START_LNG_COLUMN: lng_conversion.convert_lng_positive_in_west(
        TORNADO_TABLE_NO_INVALID_ROWS[tornado_io.START_LNG_COLUMN].values),
    tornado_io.END_LNG_COLUMN: lng_conversion.convert_lng_positive_in_west(
        TORNADO_TABLE_NO_INVALID_ROWS[tornado_io.END_LNG_COLUMN].values)
})

# The following constants are used to test interp_tornadoes_along_tracks.
THESE_START_TIMES_UNIX_SEC = numpy.array([1, 5, 5, 5, 60], dtype=int)
THESE_END_TIMES_UNIX_SEC = numpy.array([11, 10, 12, 14, 60], dtype=int)

THESE_START_LATITUDES_DEG = numpy.array([59.5, 61, 51, 49, 89])
THESE_END_LATITUDES_DEG = numpy.array([59.5, 66, 58, 58, 89])

THESE_START_LONGITUDES_DEG = numpy.array([271, 275, 242.5, 242.5, 300])
THESE_END_LONGITUDES_DEG = numpy.array([281, 275, 242.5, 242.5, 300])
THESE_FUJITA_STRINGS = ['EF1', 'EF2', 'EF3', 'EF4', 'EF5']

TORNADO_TABLE_BEFORE_INTERP = pandas.DataFrame.from_dict({
    tornado_io.START_TIME_COLUMN: THESE_START_TIMES_UNIX_SEC,
    tornado_io.END_TIME_COLUMN: THESE_END_TIMES_UNIX_SEC,
    tornado_io.START_LAT_COLUMN: THESE_START_LATITUDES_DEG,
    tornado_io.END_LAT_COLUMN: THESE_END_LATITUDES_DEG,
    tornado_io.START_LNG_COLUMN: THESE_START_LONGITUDES_DEG,
    tornado_io.END_LNG_COLUMN: THESE_END_LONGITUDES_DEG,
    tornado_io.FUJITA_RATING_COLUMN: THESE_FUJITA_STRINGS
})

INTERP_TIME_INTERVAL_SEC = 1

THESE_TIMES_UNIX_SEC = numpy.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    5, 6, 7, 8, 9, 10,
    5, 6, 7, 8, 9, 10, 11, 12,
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    60
], dtype=int)

THESE_LATITUDES_DEG = numpy.array([
    59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 59.5,
    61, 62, 63, 64, 65, 66,
    51, 52, 53, 54, 55, 56, 57, 58,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    89
])

THESE_LONGITUDES_DEG = numpy.array([
    271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281,
    275, 275, 275, 275, 275, 275,
    242.5, 242.5, 242.5, 242.5, 242.5, 242.5, 242.5, 242.5,
    242.5, 242.5, 242.5, 242.5, 242.5, 242.5, 242.5, 242.5, 242.5, 242.5,
    300
])

THESE_UNIQUE_ID_STRINGS = [
    tornado_io.create_tornado_id(
        start_time_unix_sec=t, start_latitude_deg=y, start_longitude_deg=x
    ) for t, x, y in
    zip(THESE_START_TIMES_UNIX_SEC, THESE_START_LONGITUDES_DEG,
        THESE_START_LATITUDES_DEG)
]

THESE_ID_STRINGS = (
    [THESE_UNIQUE_ID_STRINGS[0]] * 11 + [THESE_UNIQUE_ID_STRINGS[1]] * 6 +
    [THESE_UNIQUE_ID_STRINGS[2]] * 8 + [THESE_UNIQUE_ID_STRINGS[3]] * 10 +
    [THESE_UNIQUE_ID_STRINGS[4]] * 1
)

THESE_FUJITA_STRINGS = (
    [THESE_FUJITA_STRINGS[0]] * 11 + [THESE_FUJITA_STRINGS[1]] * 6 +
    [THESE_FUJITA_STRINGS[2]] * 8 + [THESE_FUJITA_STRINGS[3]] * 10 +
    [THESE_FUJITA_STRINGS[4]] * 1
)

TORNADO_TABLE_AFTER_INTERP = pandas.DataFrame.from_dict({
    tornado_io.TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    tornado_io.LATITUDE_COLUMN: THESE_LATITUDES_DEG,
    tornado_io.LONGITUDE_COLUMN: THESE_LONGITUDES_DEG,
    tornado_io.TORNADO_ID_COLUMN: THESE_ID_STRINGS,
    tornado_io.FUJITA_RATING_COLUMN: THESE_FUJITA_STRINGS
})

# The following constants are used to test find_processed_file.
TORNADO_DIRECTORY_NAME = 'tornado_reports'
YEAR = 4055
TORNADO_FILE_NAME = 'tornado_reports/tornado_reports_4055.csv'


class TornadoIoTests(unittest.TestCase):
    """Each method is a unit test for tornado_io.py."""

    def test_is_valid_fujita_rating_f_all_caps(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, input is a valid F-scale rating with ALL CAPS.
        """

        self.assertTrue(
            tornado_io._is_valid_fujita_rating(F_SCALE_RATING_ALL_CAPS)
        )

    def test_is_valid_fujita_rating_f_no_caps(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, input is a valid F-scale rating with no caps.
        """

        self.assertTrue(
            tornado_io._is_valid_fujita_rating(F_SCALE_RATING_NO_CAPS)
        )

    def test_is_valid_fujita_rating_f_leading_zero(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, F-scale rating has leading zero.
        """

        self.assertTrue(
            tornado_io._is_valid_fujita_rating(F_SCALE_RATING_LEADING_ZERO)
        )

    def test_is_valid_fujita_rating_f_too_many_letters(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, F-scale rating has too many letters.
        """

        self.assertFalse(
            tornado_io._is_valid_fujita_rating(F_SCALE_RATING_TOO_MANY_LETTERS)
        )

    def test_is_valid_fujita_rating_f_too_low(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, F-scale rating is too low.
        """

        self.assertFalse(
            tornado_io._is_valid_fujita_rating(F_SCALE_RATING_TOO_LOW)
        )

    def test_is_valid_fujita_rating_f_too_high(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, F-scale rating is too high.
        """

        self.assertFalse(
            tornado_io._is_valid_fujita_rating(F_SCALE_RATING_TOO_HIGH)
        )

    def test_is_valid_fujita_rating_ef_all_caps(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, input is a valid EF-scale rating with ALL CAPS.
        """

        self.assertTrue(
            tornado_io._is_valid_fujita_rating(EF_SCALE_RATING_ALL_CAPS)
        )

    def test_is_valid_fujita_rating_ef_some_caps(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, input is a valid EF-scale rating with SoMe CaPs.
        """

        self.assertTrue(
            tornado_io._is_valid_fujita_rating(EF_SCALE_RATING_SOME_CAPS)
        )

    def test_is_valid_fujita_rating_ef_no_caps(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, input is a valid EF-scale rating with no caps.
        """

        self.assertTrue(
            tornado_io._is_valid_fujita_rating(EF_SCALE_RATING_NO_CAPS)
        )

    def test_is_valid_fujita_rating_ef_leading_zero(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, EF-scale rating has leading zero.
        """

        self.assertTrue(
            tornado_io._is_valid_fujita_rating(EF_SCALE_RATING_LEADING_ZERO)
        )

    def test_is_valid_fujita_rating_ef_too_many_letters(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, EF-scale rating has too many letters.
        """

        self.assertFalse(tornado_io._is_valid_fujita_rating(
            EF_SCALE_RATING_TOO_MANY_LETTERS
        ))

    def test_is_valid_fujita_rating_ef_too_low(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, EF-scale rating is too low.
        """

        self.assertFalse(
            tornado_io._is_valid_fujita_rating(EF_SCALE_RATING_TOO_LOW)
        )

    def test_is_valid_fujita_rating_ef_too_high(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, EF-scale rating is too high.
        """

        self.assertFalse(
            tornado_io._is_valid_fujita_rating(EF_SCALE_RATING_TOO_HIGH)
        )

    def test_create_tornado_id(self):
        """Ensures correct output from create_tornado_id."""

        this_id_string = tornado_io.create_tornado_id(
            start_time_unix_sec=TORNADO_START_TIME_UNIX_SEC,
            start_latitude_deg=TORNADO_START_LATITUDE_DEG,
            start_longitude_deg=TORNADO_START_LONGITUDE_DEG)

        self.assertTrue(this_id_string == TORNADO_ID_STRING)

    def test_remove_invalid_reports(self):
        """Ensures correct output from remove_invalid_reports."""

        this_new_table = tornado_io.remove_invalid_reports(
            copy.deepcopy(TORNADO_TABLE_WITH_INVALID_ROWS)
        )

        self.assertTrue(this_new_table.equals(TORNADO_TABLE_NO_INVALID_ROWS))

    def test_interp_tornadoes_along_tracks(self):
        """Ensures correct output from interp_tornadoes_along_tracks."""

        this_tornado_table = tornado_io.interp_tornadoes_along_tracks(
            tornado_table=copy.deepcopy(TORNADO_TABLE_BEFORE_INTERP),
            interp_time_interval_sec=INTERP_TIME_INTERVAL_SEC)

        actual_columns = list(this_tornado_table)
        expected_columns = list(TORNADO_TABLE_AFTER_INTERP)
        self.assertTrue(set(actual_columns) == set(expected_columns))

        exact_columns = [
            tornado_io.TIME_COLUMN, tornado_io.TORNADO_ID_COLUMN,
            tornado_io.FUJITA_RATING_COLUMN
        ]

        for this_column in actual_columns:
            if this_column in exact_columns:
                self.assertTrue(numpy.array_equal(
                    this_tornado_table[this_column].values,
                    TORNADO_TABLE_AFTER_INTERP[this_column].values
                ))
            else:
                self.assertTrue(numpy.allclose(
                    this_tornado_table[this_column].values,
                    TORNADO_TABLE_AFTER_INTERP[this_column].values,
                    atol=TOLERANCE
                ))

    def test_segments_to_tornadoes(self):
        """Ensures correct output from segments_to_tornadoes."""

        this_actual_table = tornado_io.segments_to_tornadoes(
            TORNADO_TABLE_AFTER_INTERP)

        this_actual_table.sort_values(
            tornado_io.FUJITA_RATING_COLUMN, axis=0, ascending=True,
            inplace=True)

        this_expected_table = TORNADO_TABLE_BEFORE_INTERP.sort_values(
            tornado_io.FUJITA_RATING_COLUMN, axis=0, ascending=True,
            inplace=False)

        actual_columns = set(list(this_actual_table))
        expected_columns = set(list(this_expected_table))
        self.assertTrue(expected_columns.issubset(actual_columns))

        exact_columns = [
            tornado_io.START_TIME_COLUMN, tornado_io.END_TIME_COLUMN,
            tornado_io.FUJITA_RATING_COLUMN
        ]

        for this_column in expected_columns:
            if this_column in exact_columns:
                self.assertTrue(numpy.array_equal(
                    this_actual_table[this_column].values,
                    this_expected_table[this_column].values
                ))
            else:
                self.assertTrue(numpy.allclose(
                    this_actual_table[this_column].values,
                    this_expected_table[this_column].values, atol=TOLERANCE
                ))

    def test_find_processed_file(self):
        """Ensures correct output from find_processed_file."""

        this_file_name = tornado_io.find_processed_file(
            directory_name=TORNADO_DIRECTORY_NAME, year=YEAR,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == TORNADO_FILE_NAME)


if __name__ == '__main__':
    unittest.main()
