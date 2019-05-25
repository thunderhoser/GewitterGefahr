"""Unit tests for tornado_io.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_io import tornado_io
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion

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
            tornado_io._is_valid_fujita_rating(F_SCALE_RATING_ALL_CAPS))

    def test_is_valid_fujita_rating_f_no_caps(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, input is a valid F-scale rating with no caps.
        """

        self.assertTrue(
            tornado_io._is_valid_fujita_rating(F_SCALE_RATING_NO_CAPS))

    def test_is_valid_fujita_rating_f_leading_zero(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, F-scale rating has leading zero.
        """

        self.assertTrue(
            tornado_io._is_valid_fujita_rating(F_SCALE_RATING_LEADING_ZERO))

    def test_is_valid_fujita_rating_f_too_many_letters(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, F-scale rating has too many letters.
        """

        self.assertFalse(
            tornado_io._is_valid_fujita_rating(F_SCALE_RATING_TOO_MANY_LETTERS))

    def test_is_valid_fujita_rating_f_too_low(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, F-scale rating is too low.
        """

        self.assertFalse(
            tornado_io._is_valid_fujita_rating(F_SCALE_RATING_TOO_LOW))

    def test_is_valid_fujita_rating_f_too_high(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, F-scale rating is too high.
        """

        self.assertFalse(
            tornado_io._is_valid_fujita_rating(F_SCALE_RATING_TOO_HIGH))

    def test_is_valid_fujita_rating_ef_all_caps(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, input is a valid EF-scale rating with ALL CAPS.
        """

        self.assertTrue(
            tornado_io._is_valid_fujita_rating(EF_SCALE_RATING_ALL_CAPS))

    def test_is_valid_fujita_rating_ef_some_caps(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, input is a valid EF-scale rating with SoMe CaPs.
        """

        self.assertTrue(
            tornado_io._is_valid_fujita_rating(EF_SCALE_RATING_SOME_CAPS))

    def test_is_valid_fujita_rating_ef_no_caps(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, input is a valid EF-scale rating with no caps.
        """

        self.assertTrue(
            tornado_io._is_valid_fujita_rating(EF_SCALE_RATING_NO_CAPS))

    def test_is_valid_fujita_rating_ef_leading_zero(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, EF-scale rating has leading zero.
        """

        self.assertTrue(
            tornado_io._is_valid_fujita_rating(EF_SCALE_RATING_LEADING_ZERO))

    def test_is_valid_fujita_rating_ef_too_many_letters(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, EF-scale rating has too many letters.
        """

        self.assertFalse(tornado_io._is_valid_fujita_rating(
            EF_SCALE_RATING_TOO_MANY_LETTERS))

    def test_is_valid_fujita_rating_ef_too_low(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, EF-scale rating is too low.
        """

        self.assertFalse(
            tornado_io._is_valid_fujita_rating(EF_SCALE_RATING_TOO_LOW))

    def test_is_valid_fujita_rating_ef_too_high(self):
        """Ensures correct output from _is_valid_fujita_rating.

        In this case, EF-scale rating is too high.
        """

        self.assertFalse(
            tornado_io._is_valid_fujita_rating(EF_SCALE_RATING_TOO_HIGH))

    def test_remove_invalid_reports(self):
        """Ensures correct output from remove_invalid_reports."""

        this_orig_table = copy.deepcopy(TORNADO_TABLE_WITH_INVALID_ROWS)
        this_new_table = tornado_io.remove_invalid_reports(this_orig_table)
        self.assertTrue(this_new_table.equals(TORNADO_TABLE_NO_INVALID_ROWS))

    def test_find_processed_file(self):
        """Ensures correct output from find_processed_file."""

        this_file_name = tornado_io.find_processed_file(
            directory_name=TORNADO_DIRECTORY_NAME, year=YEAR,
            raise_error_if_missing=False)
        self.assertTrue(this_file_name == TORNADO_FILE_NAME)


if __name__ == '__main__':
    unittest.main()
