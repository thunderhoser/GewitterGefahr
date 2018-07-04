"""Unit tests for storm_tracking_io.py."""

import unittest
import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

VALID_TIME_UNIX_SEC = 1507167848
VALID_SPC_DATE_STRING = '20171004'
TRACKING_SCALE_METRES2 = 5e7

TOP_SEGMOTION_DIR_NAME = 'segmotion'
TOP_PROBSEVERE_DIR_NAME = 'probSevere'
SEGMOTION_FILE_NAME = (
    'segmotion/2017/20171004/scale_50000000m2/'
    'storm-tracking_segmotion_2017-10-05-014408.p')
PROBSEVERE_FILE_NAME = (
    'probSevere/2017/20171004/scale_50000000m2/'
    'storm-tracking_probSevere_2017-10-05-014408.p')

YEARS = numpy.array([4055, 4056], dtype=int)
MONTHS = numpy.array([12, 1, 2], dtype=int)
HOURS = numpy.array([3], dtype=int)

GLOB_PATTERNS_FOR_YEARS = [
    'segmotion/4055/4055[0-1][0-9][0-3][0-9]/scale_50000000m2/storm-tracking_'
    'segmotion_4055-[0-1][0-9]-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p',
    'segmotion/4054/40541231/scale_50000000m2/storm-tracking_segmotion_'
    '4055-[0-1][0-9]-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p',
    'segmotion/4056/4056[0-1][0-9][0-3][0-9]/scale_50000000m2/storm-tracking_'
    'segmotion_4056-[0-1][0-9]-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p',
    'segmotion/4055/40551231/scale_50000000m2/storm-tracking_segmotion_'
    '4056-[0-1][0-9]-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p',
]
GLOB_PATTERNS_FOR_HOURS = [
    'segmotion/[0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]/'
    'scale_50000000m2/storm-tracking_segmotion_'
    '[0-9][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9]-03[0-5][0-9][0-5][0-9].p'
]
GLOB_PATTERNS_FOR_YEARS_MONTHS = [
    'segmotion/4055/405512[0-3][0-9]/scale_50000000m2/storm-tracking_'
    'segmotion_4055-12-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p',
    'segmotion/4055/40551130/scale_50000000m2/storm-tracking_'
    'segmotion_4055-12-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p',
    'segmotion/4055/405501[0-3][0-9]/scale_50000000m2/storm-tracking_'
    'segmotion_4055-01-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p',
    'segmotion/4054/40541231/scale_50000000m2/storm-tracking_'
    'segmotion_4055-01-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p',
    'segmotion/4055/405502[0-3][0-9]/scale_50000000m2/storm-tracking_'
    'segmotion_4055-02-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p',
    'segmotion/4055/40550131/scale_50000000m2/storm-tracking_'
    'segmotion_4055-02-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p',
    'segmotion/4056/405612[0-3][0-9]/scale_50000000m2/storm-tracking_'
    'segmotion_4056-12-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p',
    'segmotion/4056/40561130/scale_50000000m2/storm-tracking_'
    'segmotion_4056-12-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p',
    'segmotion/4056/405601[0-3][0-9]/scale_50000000m2/storm-tracking_'
    'segmotion_4056-01-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p',
    'segmotion/4055/40551231/scale_50000000m2/storm-tracking_'
    'segmotion_4056-01-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p',
    'segmotion/4056/405602[0-3][0-9]/scale_50000000m2/storm-tracking_'
    'segmotion_4056-02-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p',
    'segmotion/4056/40560131/scale_50000000m2/storm-tracking_'
    'segmotion_4056-02-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p'
]
GLOB_PATTERNS_FOR_YEARS_MONTHS_HOURS = [
    'segmotion/4055/405512[0-3][0-9]/scale_50000000m2/storm-tracking_'
    'segmotion_4055-12-[0-3][0-9]-03[0-5][0-9][0-5][0-9].p',
    'segmotion/4055/40551130/scale_50000000m2/storm-tracking_'
    'segmotion_4055-12-[0-3][0-9]-03[0-5][0-9][0-5][0-9].p',
    'segmotion/4055/405501[0-3][0-9]/scale_50000000m2/storm-tracking_'
    'segmotion_4055-01-[0-3][0-9]-03[0-5][0-9][0-5][0-9].p',
    'segmotion/4054/40541231/scale_50000000m2/storm-tracking_'
    'segmotion_4055-01-[0-3][0-9]-03[0-5][0-9][0-5][0-9].p',
    'segmotion/4055/405502[0-3][0-9]/scale_50000000m2/storm-tracking_'
    'segmotion_4055-02-[0-3][0-9]-03[0-5][0-9][0-5][0-9].p',
    'segmotion/4055/40550131/scale_50000000m2/storm-tracking_'
    'segmotion_4055-02-[0-3][0-9]-03[0-5][0-9][0-5][0-9].p',
    'segmotion/4056/405612[0-3][0-9]/scale_50000000m2/storm-tracking_'
    'segmotion_4056-12-[0-3][0-9]-03[0-5][0-9][0-5][0-9].p',
    'segmotion/4056/40561130/scale_50000000m2/storm-tracking_'
    'segmotion_4056-12-[0-3][0-9]-03[0-5][0-9][0-5][0-9].p',
    'segmotion/4056/405601[0-3][0-9]/scale_50000000m2/storm-tracking_'
    'segmotion_4056-01-[0-3][0-9]-03[0-5][0-9][0-5][0-9].p',
    'segmotion/4055/40551231/scale_50000000m2/storm-tracking_'
    'segmotion_4056-01-[0-3][0-9]-03[0-5][0-9][0-5][0-9].p',
    'segmotion/4056/405602[0-3][0-9]/scale_50000000m2/storm-tracking_'
    'segmotion_4056-02-[0-3][0-9]-03[0-5][0-9][0-5][0-9].p',
    'segmotion/4056/40560131/scale_50000000m2/storm-tracking_'
    'segmotion_4056-02-[0-3][0-9]-03[0-5][0-9][0-5][0-9].p'
]

GLOB_PATTERN_FOR_SPC_DATE = (
    'segmotion/2017/20171004/scale_50000000m2/storm-tracking_segmotion_'
    '[0-9][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9]'
    '.p')


class StormTrackingIoTests(unittest.TestCase):
    """Each method is a unit test for storm_tracking_io.py."""

    def test_get_previous_month_mid_year(self):
        """Ensures correct output from _get_previous_month.

        In this case, current month is in the middle of the year.
        """

        this_current_month = 6
        this_current_year = 4000
        (this_previous_month, this_previous_year
        ) = tracking_io._get_previous_month(
            month=this_current_month, year=this_current_year)

        self.assertTrue(this_previous_month == this_current_month - 1)
        self.assertTrue(this_previous_year == this_current_year)

    def test_get_previous_month_january(self):
        """Ensures correct output from _get_previous_month.

        In this case, current month is January.
        """

        this_current_month = 1
        this_current_year = 4000
        (this_previous_month, this_previous_year
        ) = tracking_io._get_previous_month(
            month=this_current_month, year=this_current_year)

        self.assertTrue(this_previous_month == 12)
        self.assertTrue(this_previous_year == this_current_year - 1)

    def test_get_num_days_in_month_april(self):
        """Ensures correct output from _get_num_days_in_month.

        In this case the month is April.
        """

        this_num_days_in_month = tracking_io._get_num_days_in_month(
            month=4, year=4000)
        self.assertTrue(this_num_days_in_month == 30)

    def test_get_num_days_in_month_may(self):
        """Ensures correct output from _get_num_days_in_month.

        In this case the month is May.
        """

        this_num_days_in_month = tracking_io._get_num_days_in_month(
            month=5, year=4000)
        self.assertTrue(this_num_days_in_month == 31)

    def test_get_num_days_in_month_feb2000(self):
        """Ensures correct output from _get_num_days_in_month.

        In this case the month is Feb 2000 (leap year).
        """

        this_num_days_in_month = tracking_io._get_num_days_in_month(
            month=2, year=2000)
        self.assertTrue(this_num_days_in_month == 29)

    def test_get_num_days_in_month_feb2100(self):
        """Ensures correct output from _get_num_days_in_month.

        In this case the month is Feb 2100 (non-leap year).
        """

        this_num_days_in_month = tracking_io._get_num_days_in_month(
            month=2, year=2100)
        self.assertTrue(this_num_days_in_month == 28)

    def test_find_processed_file_segmotion(self):
        """Ensures correct output from find_processed_file.

        In this case, data source is segmotion.
        """

        this_processed_file_name = tracking_io.find_processed_file(
            top_processed_dir_name=TOP_SEGMOTION_DIR_NAME,
            tracking_scale_metres2=TRACKING_SCALE_METRES2,
            data_source=tracking_utils.SEGMOTION_SOURCE_ID,
            unix_time_sec=VALID_TIME_UNIX_SEC,
            spc_date_string=VALID_SPC_DATE_STRING,
            raise_error_if_missing=False)

        self.assertTrue(this_processed_file_name == SEGMOTION_FILE_NAME)

    def test_find_processed_file_probsevere(self):
        """Ensures correct output from find_processed_file.

        In this case, data source is probSevere.
        """

        this_processed_file_name = tracking_io.find_processed_file(
            top_processed_dir_name=TOP_PROBSEVERE_DIR_NAME,
            tracking_scale_metres2=TRACKING_SCALE_METRES2,
            data_source=tracking_utils.PROBSEVERE_SOURCE_ID,
            unix_time_sec=VALID_TIME_UNIX_SEC,
            raise_error_if_missing=False)

        self.assertTrue(this_processed_file_name == PROBSEVERE_FILE_NAME)

    def test_find_processed_files_at_times_years(self):
        """Ensures correct output from find_processed_files_at_times.

        In this case, looking only for specific years.
        """

        _, these_glob_patterns = tracking_io.find_processed_files_at_times(
            top_processed_dir_name=TOP_SEGMOTION_DIR_NAME,
            tracking_scale_metres2=TRACKING_SCALE_METRES2,
            data_source=tracking_utils.SEGMOTION_SOURCE_ID,
            years=YEARS, months=None, hours=None, raise_error_if_missing=False)

        self.assertTrue(set(these_glob_patterns) ==
                        set(GLOB_PATTERNS_FOR_YEARS))

    def test_find_processed_files_at_times_hours(self):
        """Ensures correct output from find_processed_files_at_times.

        In this case, looking only for specific hours.
        """

        _, these_glob_patterns = tracking_io.find_processed_files_at_times(
            top_processed_dir_name=TOP_SEGMOTION_DIR_NAME,
            tracking_scale_metres2=TRACKING_SCALE_METRES2,
            data_source=tracking_utils.SEGMOTION_SOURCE_ID,
            years=None, months=None, hours=HOURS, raise_error_if_missing=False)

        self.assertTrue(set(these_glob_patterns) ==
                        set(GLOB_PATTERNS_FOR_HOURS))

    def test_find_processed_files_at_times_years_months(self):
        """Ensures correct output from find_processed_files_at_times.

        In this case, looking for specific years/months but not hours.
        """

        _, these_glob_patterns = tracking_io.find_processed_files_at_times(
            top_processed_dir_name=TOP_SEGMOTION_DIR_NAME,
            tracking_scale_metres2=TRACKING_SCALE_METRES2,
            data_source=tracking_utils.SEGMOTION_SOURCE_ID,
            years=YEARS, months=MONTHS, hours=None,
            raise_error_if_missing=False)

        self.assertTrue(set(these_glob_patterns) ==
                        set(GLOB_PATTERNS_FOR_YEARS_MONTHS))

    def test_find_processed_files_at_times_years_hours(self):
        """Ensures correct output from find_processed_files_at_times.

        In this case, looking for specific years/months/hours.
        """

        _, these_glob_patterns = tracking_io.find_processed_files_at_times(
            top_processed_dir_name=TOP_SEGMOTION_DIR_NAME,
            tracking_scale_metres2=TRACKING_SCALE_METRES2,
            data_source=tracking_utils.SEGMOTION_SOURCE_ID,
            years=YEARS, months=MONTHS, hours=HOURS,
            raise_error_if_missing=False)

        self.assertTrue(set(these_glob_patterns) ==
                        set(GLOB_PATTERNS_FOR_YEARS_MONTHS_HOURS))

    def test_find_processed_files_one_spc_date(self):
        """Ensures correct output from find_processed_files_one_spc_date."""

        _, this_glob_pattern = tracking_io.find_processed_files_one_spc_date(
            top_processed_dir_name=TOP_SEGMOTION_DIR_NAME,
            tracking_scale_metres2=TRACKING_SCALE_METRES2,
            data_source=tracking_utils.SEGMOTION_SOURCE_ID,
            spc_date_string=VALID_SPC_DATE_STRING, raise_error_if_missing=False)

        self.assertTrue(
            this_glob_pattern == GLOB_PATTERN_FOR_SPC_DATE)


if __name__ == '__main__':
    unittest.main()
