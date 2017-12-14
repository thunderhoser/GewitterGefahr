"""Unit tests for time_conversion.py."""

import unittest
from gewittergefahr.gg_utils import time_conversion

TIME_FORMAT_YEAR = '%Y'
TIME_FORMAT_NUMERIC_MONTH = '%m'
TIME_FORMAT_3LETTER_MONTH = '%b'
TIME_FORMAT_YEAR_MONTH = '%Y-%m'
TIME_FORMAT_DAY_OF_MONTH = '%d'
TIME_FORMAT_DATE = '%Y-%m-%d'
TIME_FORMAT_HOUR = '%Y-%m-%d-%H00'
TIME_FORMAT_MINUTE = '%Y-%m-%d-%H%M'
TIME_FORMAT_SECOND = '%Y-%m-%d-%H%M%S'

TIME_STRING_YEAR = '2017'
TIME_STRING_NUMERIC_MONTH = '09'
TIME_STRING_3LETTER_MONTH = 'Sep'
TIME_STRING_YEAR_MONTH = '2017-09'
TIME_STRING_DAY_OF_MONTH = '26'
TIME_STRING_DATE = '2017-09-26'
TIME_STRING_HOUR = '2017-09-26-0500'
TIME_STRING_MINUTE = '2017-09-26-0520'
TIME_STRING_SECOND = '2017-09-26-052033'

UNIX_TIME_YEAR_SEC = 1483228800
UNIX_TIME_MONTH_SEC = 1504224000
UNIX_TIME_DATE_SEC = 1506384000
UNIX_TIME_HOUR_SEC = 1506402000
UNIX_TIME_MINUTE_SEC = 1506403200
UNIX_TIME_SEC = 1506403233

START_TIME_SEP2017_UNIX_SEC = 1504224000
END_TIME_SEP2017_UNIX_SEC = 1506815999
START_TIME_2017_UNIX_SEC = 1483228800
END_TIME_2017_UNIX_SEC = 1514764799

TIME_1200UTC_SPC_DATE_UNIX_SEC = 1506340800
TIME_0000UTC_SPC_DATE_UNIX_SEC = 1506384000
TIME_115959UTC_SPC_DATE_UNIX_SEC = 1506427199
SPC_DATE_UNIX_SEC = 1506362400
SPC_DATE_STRING = '20170925'

TIME_115959UTC_BEFORE_DATE_UNIX_SEC = 1506340799
TIME_1200UTC_AFTER_DATE_UNIX_SEC = 1506427200


class TimeConversionTests(unittest.TestCase):
    """Each method is a unit test for time_conversion.py."""

    def test_string_to_unix_sec_year(self):
        """Ensures correctness of string_to_unix_sec; string = year only."""

        this_time_unix_sec = time_conversion.string_to_unix_sec(
            TIME_STRING_YEAR, TIME_FORMAT_YEAR)
        self.assertTrue(this_time_unix_sec == UNIX_TIME_YEAR_SEC)

    def test_string_to_unix_sec_year_month(self):
        """Ensures correctness of string_to_unix_sec; string = year-month."""

        this_time_unix_sec = time_conversion.string_to_unix_sec(
            TIME_STRING_YEAR_MONTH, TIME_FORMAT_YEAR_MONTH)
        self.assertTrue(this_time_unix_sec == UNIX_TIME_MONTH_SEC)

    def test_string_to_unix_sec_date(self):
        """Ensures correctness of string_to_unix_sec; string = full date."""

        this_time_unix_sec = time_conversion.string_to_unix_sec(
            TIME_STRING_DATE, TIME_FORMAT_DATE)
        self.assertTrue(this_time_unix_sec == UNIX_TIME_DATE_SEC)

    def test_string_to_unix_sec_hour(self):
        """Ensures correctness of string_to_unix_sec; string = full hour."""

        this_time_unix_sec = time_conversion.string_to_unix_sec(
            TIME_STRING_HOUR, TIME_FORMAT_HOUR)
        self.assertTrue(this_time_unix_sec == UNIX_TIME_HOUR_SEC)

    def test_string_to_unix_sec_minute(self):
        """Ensures correctness of string_to_unix_sec; string = full minute."""

        this_time_unix_sec = time_conversion.string_to_unix_sec(
            TIME_STRING_MINUTE, TIME_FORMAT_MINUTE)
        self.assertTrue(this_time_unix_sec == UNIX_TIME_MINUTE_SEC)

    def test_string_to_unix_sec_second(self):
        """Ensures correctness of string_to_unix_sec; string = full second."""

        this_time_unix_sec = time_conversion.string_to_unix_sec(
            TIME_STRING_SECOND, TIME_FORMAT_SECOND)
        self.assertTrue(this_time_unix_sec == UNIX_TIME_SEC)

    def test_unix_sec_to_string_year(self):
        """Ensures correctness of unix_sec_to_string; string = year only."""

        this_time_string = time_conversion.unix_sec_to_string(
            UNIX_TIME_SEC, TIME_FORMAT_YEAR)
        self.assertTrue(this_time_string == TIME_STRING_YEAR)

    def test_unix_sec_to_string_numeric_month(self):
        """Ensures correctness of unix_sec_to_string; string = numeric month."""

        this_time_string = time_conversion.unix_sec_to_string(
            UNIX_TIME_SEC, TIME_FORMAT_NUMERIC_MONTH)
        self.assertTrue(this_time_string == TIME_STRING_NUMERIC_MONTH)

    def test_unix_sec_to_string_3letter_month(self):
        """Ensures correctness of unix_sec_to_string; string = 3-lttr month."""

        this_time_string = time_conversion.unix_sec_to_string(
            UNIX_TIME_SEC, TIME_FORMAT_3LETTER_MONTH)
        self.assertTrue(this_time_string == TIME_STRING_3LETTER_MONTH)

    def test_unix_sec_to_string_year_month(self):
        """Ensures correctness of unix_sec_to_string; string = year-month."""

        this_time_string = time_conversion.unix_sec_to_string(
            UNIX_TIME_SEC, TIME_FORMAT_YEAR_MONTH)
        self.assertTrue(this_time_string == TIME_STRING_YEAR_MONTH)

    def test_unix_sec_to_string_day_of_month(self):
        """Ensures correctness of unix_sec_to_string; string = day of month."""

        this_time_string = time_conversion.unix_sec_to_string(
            UNIX_TIME_SEC, TIME_FORMAT_DAY_OF_MONTH)
        self.assertTrue(this_time_string == TIME_STRING_DAY_OF_MONTH)

    def test_unix_sec_to_string_date(self):
        """Ensures correctness of unix_sec_to_string; string = full date."""

        this_time_string = time_conversion.unix_sec_to_string(
            UNIX_TIME_SEC, TIME_FORMAT_DATE)
        self.assertTrue(this_time_string == TIME_STRING_DATE)

    def test_unix_sec_to_string_hour(self):
        """Ensures correctness of unix_sec_to_string; string = full hour."""

        this_time_string = time_conversion.unix_sec_to_string(
            UNIX_TIME_SEC, TIME_FORMAT_HOUR)
        self.assertTrue(this_time_string == TIME_STRING_HOUR)

    def test_unix_sec_to_string_minute(self):
        """Ensures correctness of unix_sec_to_string; string = full minute."""

        this_time_string = time_conversion.unix_sec_to_string(
            UNIX_TIME_SEC, TIME_FORMAT_MINUTE)
        self.assertTrue(this_time_string == TIME_STRING_MINUTE)

    def test_unix_sec_to_string_second(self):
        """Ensures correctness of unix_sec_to_string; string = full second."""

        this_time_string = time_conversion.unix_sec_to_string(
            UNIX_TIME_SEC, TIME_FORMAT_SECOND)
        self.assertTrue(this_time_string == TIME_STRING_SECOND)

    def test_time_to_spc_date_unix_sec_1200utc(self):
        """Ensures correctness of time_to_spc_date_unix_sec; time = 1200 UTC."""

        this_spc_date_unix_sec = time_conversion.time_to_spc_date_unix_sec(
            TIME_1200UTC_SPC_DATE_UNIX_SEC)
        self.assertTrue(this_spc_date_unix_sec == SPC_DATE_UNIX_SEC)

    def test_time_to_spc_date_unix_sec_0000utc(self):
        """Ensures correctness of time_to_spc_date_unix_sec; time = 0000 UTC."""

        this_spc_date_unix_sec = time_conversion.time_to_spc_date_unix_sec(
            TIME_0000UTC_SPC_DATE_UNIX_SEC)
        self.assertTrue(this_spc_date_unix_sec == SPC_DATE_UNIX_SEC)

    def test_time_to_spc_date_unix_sec_115959utc(self):
        """Ensures crrctness of time_to_spc_date_unix_sec; time = 115959 UTC."""

        this_spc_date_unix_sec = time_conversion.time_to_spc_date_unix_sec(
            TIME_115959UTC_SPC_DATE_UNIX_SEC)
        self.assertTrue(this_spc_date_unix_sec == SPC_DATE_UNIX_SEC)

    def test_time_to_spc_date_string_1200utc(self):
        """Ensures correctness of time_to_spc_date_string; time = 1200 UTC."""

        this_spc_date_string = time_conversion.time_to_spc_date_string(
            TIME_1200UTC_SPC_DATE_UNIX_SEC)
        self.assertTrue(this_spc_date_string == SPC_DATE_STRING)

    def test_time_to_spc_date_string_0000utc(self):
        """Ensures correctness of time_to_spc_date_string; time = 0000 UTC."""

        this_spc_date_string = time_conversion.time_to_spc_date_string(
            TIME_0000UTC_SPC_DATE_UNIX_SEC)
        self.assertTrue(this_spc_date_string == SPC_DATE_STRING)

    def test_time_to_spc_date_string_115959utc(self):
        """Ensures correctness of time_to_spc_date_string; time = 115959 UTC."""

        this_spc_date_string = time_conversion.time_to_spc_date_string(
            TIME_115959UTC_SPC_DATE_UNIX_SEC)
        self.assertTrue(this_spc_date_string == SPC_DATE_STRING)

    def test_spc_date_string_to_unix_sec(self):
        """Ensures correct output from spc_date_string_to_unix_sec."""

        this_spc_date_unix_sec = time_conversion.spc_date_string_to_unix_sec(
            SPC_DATE_STRING)
        self.assertTrue(this_spc_date_unix_sec == SPC_DATE_UNIX_SEC)

    def test_is_time_in_spc_date_beginning(self):
        """Ensures correct output from is_time_in_spc_date.

        In this case, time is at beginning of SPC date.
        """

        self.assertTrue(time_conversion.is_time_in_spc_date(
            TIME_1200UTC_SPC_DATE_UNIX_SEC, SPC_DATE_STRING))

    def test_is_time_in_spc_date_middle(self):
        """Ensures correct output from is_time_in_spc_date.

        In this case, time is in middle of SPC date.
        """

        self.assertTrue(time_conversion.is_time_in_spc_date(
            TIME_0000UTC_SPC_DATE_UNIX_SEC, SPC_DATE_STRING))

    def test_is_time_in_spc_date_end(self):
        """Ensures correct output from is_time_in_spc_date.

        In this case, time is at end of SPC date.
        """

        self.assertTrue(time_conversion.is_time_in_spc_date(
            TIME_115959UTC_SPC_DATE_UNIX_SEC, SPC_DATE_STRING))

    def test_is_time_in_spc_date_before(self):
        """Ensures correct output from is_time_in_spc_date.

        In this case, time is before SPC date.
        """

        self.assertFalse(time_conversion.is_time_in_spc_date(
            TIME_115959UTC_BEFORE_DATE_UNIX_SEC, SPC_DATE_STRING))

    def test_is_time_in_spc_date_after(self):
        """Ensures correct output from is_time_in_spc_date.

        In this case, time is after SPC date.
        """

        self.assertFalse(time_conversion.is_time_in_spc_date(
            TIME_1200UTC_AFTER_DATE_UNIX_SEC, SPC_DATE_STRING))

    def test_first_and_last_times_in_month(self):
        """Ensures correct output from first_and_last_times_in_month."""

        this_start_time_unix_sec, this_end_time_unix_sec = (
            time_conversion.first_and_last_times_in_month(UNIX_TIME_MONTH_SEC))
        self.assertTrue(this_start_time_unix_sec == START_TIME_SEP2017_UNIX_SEC)
        self.assertTrue(this_end_time_unix_sec == END_TIME_SEP2017_UNIX_SEC)

    def test_first_and_last_times_in_year(self):
        """Ensures correct output from first_and_last_times_in_year."""

        this_start_time_unix_sec, this_end_time_unix_sec = (
            time_conversion.first_and_last_times_in_year(2017))
        self.assertTrue(this_start_time_unix_sec == START_TIME_2017_UNIX_SEC)
        self.assertTrue(this_end_time_unix_sec == END_TIME_2017_UNIX_SEC)


if __name__ == '__main__':
    unittest.main()
