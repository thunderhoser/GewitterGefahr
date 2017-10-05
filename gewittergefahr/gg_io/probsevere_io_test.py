"""Unit tests for probsevere_io.py."""

import unittest
from gewittergefahr.gg_io import probsevere_io

UNIX_TIME_SEC = 1507181187  # 052627 5 Oct 2017
PATHLESS_RAW_FILE_NAME = 'SSEC_AWIPS_PROBSEVERE_20171005_052627.json'

TOP_FTP_DIRECTORY_NAME = '/data/storm_tracking/probSevere'
RAW_FTP_FILE_NAME = (
    '/data/storm_tracking/probSevere/SSEC_AWIPS_PROBSEVERE_20171005_052627'
    '.json')

TOP_LOCAL_DIRECTORY_NAME = '/data/storm_tracking/probSevere'
RAW_LOCAL_FILE_NAME = (
    '/data/storm_tracking/probSevere/20171005/'
    'SSEC_AWIPS_PROBSEVERE_20171005_052627.json')


class ProbsevereIoTests(unittest.TestCase):
    """Each method is a unit test for probsevere_io.py."""

    def test_get_pathless_raw_file_name(self):
        """Ensures correct output from _get_pathless_raw_file_name."""

        this_pathless_file_name = probsevere_io._get_pathless_raw_file_name(
            UNIX_TIME_SEC)
        self.assertTrue(this_pathless_file_name == PATHLESS_RAW_FILE_NAME)

    def test_get_raw_file_name_on_ftp(self):
        """Ensures correct output from get_raw_file_name_on_ftp."""

        this_raw_file_name = probsevere_io.get_raw_file_name_on_ftp(
            UNIX_TIME_SEC, TOP_FTP_DIRECTORY_NAME)
        self.assertTrue(this_raw_file_name == RAW_FTP_FILE_NAME)

    def test_get_raw_file_name_on_local_machine(self):
        """Ensures correct output from get_raw_file_name_on_local_machine."""

        this_raw_file_name = probsevere_io.find_raw_file_on_local_machine(
            unix_time_sec=UNIX_TIME_SEC,
            top_local_directory_name=TOP_LOCAL_DIRECTORY_NAME,
            raise_error_if_missing=False)
        self.assertTrue(this_raw_file_name == RAW_LOCAL_FILE_NAME)


if __name__ == '__main__':
    unittest.main()
