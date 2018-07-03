"""Unit tests for storm_tracking_io.py."""

import unittest
import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

UNIX_TIME_SEC = 1507167848
SPC_DATE_STRING = '20171004'
TRACKING_SCALE_METRES2 = 5e7

RELATIVE_SEGMOTION_DIR_NAME = '2017/20171004/scale_50000000m2'
PATHLESS_SEGMOTION_FILE_NAME = 'storm-tracking_segmotion_2017-10-05-014408.p'
TOP_PROCESSED_DIR_NAME_SEGMOTION = 'segmotion'
SEGMOTION_FILE_NAME = (
    'segmotion/2017/20171004/scale_50000000m2/'
    'storm-tracking_segmotion_2017-10-05-014408.p')

PATHLESS_PROBSEVERE_FILE_NAME = 'storm-tracking_probSevere_2017-10-05-014408.p'
RELATIVE_PROBSEVERE_DIR_NAME = '2017/20171004/scale_50000000m2'
TOP_PROCESSED_DIR_NAME_PROBSEVERE = 'probSevere'
PROBSEVERE_FILE_NAME = (
    'probSevere/2017/20171004/scale_50000000m2/'
    'storm-tracking_probSevere_2017-10-05-014408.p')

GLOB_PATTERN_FOR_SPC_DATE = (
    'segmotion/2017/20171004/scale_50000000m2/storm-tracking_segmotion_'
    '[0-9][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9]'
    '.p')

YEARS = numpy.array([4055, 4056], dtype=int)
MONTHS = numpy.array([12, 1, 2], dtype=int)
HOURS = numpy.array([3], dtype=int)

GLOB_PATTERNS_FOR_YEARS = [
    'segmotion/[0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]/'
    'scale_50000000m2/storm-tracking_segmotion_'
    '4055-[0-1][0-9]-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p',
    'segmotion/[0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]/'
    'scale_50000000m2/storm-tracking_segmotion_'
    '4056-[0-1][0-9]-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p'
]
GLOB_PATTERNS_FOR_YEARS_MONTHS = [
    'segmotion/[0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]/'
    'scale_50000000m2/storm-tracking_segmotion_'
    '4055-12-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p',
    'segmotion/[0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]/'
    'scale_50000000m2/storm-tracking_segmotion_'
    '4056-12-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p',
    'segmotion/[0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]/'
    'scale_50000000m2/storm-tracking_segmotion_'
    '4055-01-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p',
    'segmotion/[0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]/'
    'scale_50000000m2/storm-tracking_segmotion_'
    '4056-01-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p',
    'segmotion/[0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]/'
    'scale_50000000m2/storm-tracking_segmotion_'
    '4055-02-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p',
    'segmotion/[0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]/'
    'scale_50000000m2/storm-tracking_segmotion_'
    '4056-02-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9].p'
]
GLOB_PATTERNS_FOR_YEARS_MONTHS_HOURS = [
    'segmotion/[0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]/'
    'scale_50000000m2/storm-tracking_segmotion_'
    '4055-12-[0-3][0-9]-03[0-5][0-9][0-5][0-9].p',
    'segmotion/[0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]/'
    'scale_50000000m2/storm-tracking_segmotion_'
    '4056-12-[0-3][0-9]-03[0-5][0-9][0-5][0-9].p',
    'segmotion/[0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]/'
    'scale_50000000m2/storm-tracking_segmotion_'
    '4055-01-[0-3][0-9]-03[0-5][0-9][0-5][0-9].p',
    'segmotion/[0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]/'
    'scale_50000000m2/storm-tracking_segmotion_'
    '4056-01-[0-3][0-9]-03[0-5][0-9][0-5][0-9].p',
    'segmotion/[0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]/'
    'scale_50000000m2/storm-tracking_segmotion_'
    '4055-02-[0-3][0-9]-03[0-5][0-9][0-5][0-9].p',
    'segmotion/[0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]/'
    'scale_50000000m2/storm-tracking_segmotion_'
    '4056-02-[0-3][0-9]-03[0-5][0-9][0-5][0-9].p'
]


class StormTrackingIoTests(unittest.TestCase):
    """Each method is a unit test for storm_tracking_io.py."""

    def test_get_pathless_processed_file_name_segmotion(self):
        """Ensures correct output from _get_pathless_processed_file_name.

        In this case, data source is segmotion.
        """

        this_pathless_file_name = tracking_io._get_pathless_processed_file_name(
            UNIX_TIME_SEC, tracking_utils.SEGMOTION_SOURCE_ID)
        self.assertTrue(this_pathless_file_name == PATHLESS_SEGMOTION_FILE_NAME)

    def test_get_pathless_processed_file_name_probsevere(self):
        """Ensures correct output from _get_pathless_processed_file_name.

        In this case, data source is probSevere.
        """

        this_pathless_file_name = tracking_io._get_pathless_processed_file_name(
            UNIX_TIME_SEC, tracking_utils.PROBSEVERE_SOURCE_ID)
        self.assertTrue(this_pathless_file_name ==
                        PATHLESS_PROBSEVERE_FILE_NAME)

    def test_get_relative_processed_directory_segmotion(self):
        """Ensures correct output from _get_relative_processed_directory.

        In this case, data source is segmotion.
        """

        this_relative_dir_name = tracking_io._get_relative_processed_directory(
            data_source=tracking_utils.SEGMOTION_SOURCE_ID,
            spc_date_string=SPC_DATE_STRING,
            tracking_scale_metres2=TRACKING_SCALE_METRES2)
        self.assertTrue(this_relative_dir_name == RELATIVE_SEGMOTION_DIR_NAME)

    def test_get_relative_processed_directory_probsevere(self):
        """Ensures correct output from _get_relative_processed_directory.

        In this case, data source is probSevere.
        """

        this_relative_dir_name = tracking_io._get_relative_processed_directory(
            data_source=tracking_utils.PROBSEVERE_SOURCE_ID,
            unix_time_sec=UNIX_TIME_SEC,
            tracking_scale_metres2=TRACKING_SCALE_METRES2)
        self.assertTrue(this_relative_dir_name == RELATIVE_PROBSEVERE_DIR_NAME)

    def test_find_processed_file_segmotion(self):
        """Ensures correct output from find_processed_file.

        In this case, data source is segmotion.
        """

        this_processed_file_name = tracking_io.find_processed_file(
            unix_time_sec=UNIX_TIME_SEC,
            data_source=tracking_utils.SEGMOTION_SOURCE_ID,
            spc_date_string=SPC_DATE_STRING,
            top_processed_dir_name=TOP_PROCESSED_DIR_NAME_SEGMOTION,
            tracking_scale_metres2=TRACKING_SCALE_METRES2,
            raise_error_if_missing=False)
        self.assertTrue(this_processed_file_name == SEGMOTION_FILE_NAME)

    def test_find_processed_file_probSevere(self):
        """Ensures correct output from find_processed_file.

        In this case, data source is probSevere.
        """

        this_processed_file_name = tracking_io.find_processed_file(
            unix_time_sec=UNIX_TIME_SEC,
            data_source=tracking_utils.PROBSEVERE_SOURCE_ID,
            top_processed_dir_name=TOP_PROCESSED_DIR_NAME_PROBSEVERE,
            tracking_scale_metres2=TRACKING_SCALE_METRES2,
            raise_error_if_missing=False)
        self.assertTrue(this_processed_file_name == PROBSEVERE_FILE_NAME)

    def test_find_processed_files_at_times_years(self):
        """Ensures correct output from find_processed_files_at_times.

        In this case, looking only for specific years.
        """

        _, these_glob_patterns = tracking_io.find_processed_files_at_times(
            top_processed_dir_name=TOP_PROCESSED_DIR_NAME_SEGMOTION,
            tracking_scale_metres2=TRACKING_SCALE_METRES2,
            data_source=tracking_utils.SEGMOTION_SOURCE_ID,
            years=YEARS, months=None, hours=None, raise_error_if_missing=False)

        self.assertTrue(set(these_glob_patterns) ==
                        set(GLOB_PATTERNS_FOR_YEARS))

    def test_find_processed_files_at_times_years_months(self):
        """Ensures correct output from find_processed_files_at_times.

        In this case, looking for specific years/months but not hours.
        """

        _, these_glob_patterns = tracking_io.find_processed_files_at_times(
            top_processed_dir_name=TOP_PROCESSED_DIR_NAME_SEGMOTION,
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
            top_processed_dir_name=TOP_PROCESSED_DIR_NAME_SEGMOTION,
            tracking_scale_metres2=TRACKING_SCALE_METRES2,
            data_source=tracking_utils.SEGMOTION_SOURCE_ID,
            years=YEARS, months=MONTHS, hours=HOURS,
            raise_error_if_missing=False)

        self.assertTrue(set(these_glob_patterns) ==
                        set(GLOB_PATTERNS_FOR_YEARS_MONTHS_HOURS))

    def test_find_processed_files_one_spc_date(self):
        """Ensures correct output from find_processed_files_one_spc_date."""

        _, this_glob_pattern = tracking_io.find_processed_files_one_spc_date(
            top_processed_dir_name=TOP_PROCESSED_DIR_NAME_SEGMOTION,
            tracking_scale_metres2=TRACKING_SCALE_METRES2,
            data_source=tracking_utils.SEGMOTION_SOURCE_ID,
            spc_date_string=SPC_DATE_STRING, raise_error_if_missing=False)

        self.assertTrue(
            this_glob_pattern == GLOB_PATTERN_FOR_SPC_DATE)


if __name__ == '__main__':
    unittest.main()
