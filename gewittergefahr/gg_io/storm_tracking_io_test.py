"""Unit tests for storm_tracking_io.py."""

import unittest
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

UNIX_TIME_SEC = 1507167848
SPC_DATE_STRING = '20171004'
TRACKING_SCALE_METRES2 = 5e7

RELATIVE_SEGMOTION_DIR_NAME = '20171004/scale_50000000m2'
PATHLESS_SEGMOTION_FILE_NAME = 'storm-tracking_segmotion_2017-10-05-014408.p'
TOP_PROCESSED_DIR_NAME_SEGMOTION = 'segmotion'
SEGMOTION_FILE_NAME = (
    'segmotion/20171004/scale_50000000m2/'
    'storm-tracking_segmotion_2017-10-05-014408.p')

PATHLESS_PROBSEVERE_FILE_NAME = 'storm-tracking_probSevere_2017-10-05-014408.p'
RELATIVE_PROBSEVERE_DIR_NAME = '20171005/scale_50000000m2'
TOP_PROCESSED_DIR_NAME_PROBSEVERE = 'probSevere'
PROBSEVERE_FILE_NAME = (
    'probSevere/20171005/scale_50000000m2/'
    'storm-tracking_probSevere_2017-10-05-014408.p')


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


if __name__ == '__main__':
    unittest.main()
