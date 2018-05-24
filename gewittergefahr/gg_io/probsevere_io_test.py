"""Unit tests for probsevere_io.py."""

import unittest
from gewittergefahr.gg_io import probsevere_io

TOP_DIRECTORY_NAME = 'foo'
FTP_DIRECTORY_NAME = 'bar'
VALID_TIME_UNIX_SEC = 1507181187  # 052627 5 Oct 2017

PATHLESS_JSON_FILE_NAME = 'SSEC_AWIPS_PROBSEVERE_20171005_052627.json'
PATHLESS_ASCII_FILE_NAME = 'SSEC_AWIPS_PROBSEVERE_20171005_052627.ascii'

JSON_FILE_NAME = (
    'foo/201710/20171005/SSEC_AWIPS_PROBSEVERE_20171005_052627.json')
ASCII_FILE_NAME = (
    'foo/201710/20171005/SSEC_AWIPS_PROBSEVERE_20171005_052627.ascii')
JSON_FILE_NAME_ON_FTP = 'bar/SSEC_AWIPS_PROBSEVERE_20171005_052627.json'


class ProbsevereIoTests(unittest.TestCase):
    """Each method is a unit test for probsevere_io.py."""

    def test_get_pathless_raw_file_name_json(self):
        """Ensures correct output from _get_pathless_raw_file_name.

        In this case the file type is JSON.
        """

        this_pathless_file_name = probsevere_io._get_pathless_raw_file_name(
            unix_time_sec=VALID_TIME_UNIX_SEC,
            file_extension=probsevere_io.JSON_FILE_EXTENSION)
        self.assertTrue(this_pathless_file_name == PATHLESS_JSON_FILE_NAME)

    def test_get_pathless_raw_file_name_ascii(self):
        """Ensures correct output from _get_pathless_raw_file_name.

        In this case the file type is ASCII.
        """

        this_pathless_file_name = probsevere_io._get_pathless_raw_file_name(
            unix_time_sec=VALID_TIME_UNIX_SEC,
            file_extension=probsevere_io.ASCII_FILE_EXTENSION)
        self.assertTrue(this_pathless_file_name == PATHLESS_ASCII_FILE_NAME)

    def test_get_json_file_name_on_ftp(self):
        """Ensures correct output from get_json_file_name_on_ftp."""

        this_file_name = probsevere_io.get_json_file_name_on_ftp(
            unix_time_sec=VALID_TIME_UNIX_SEC,
            ftp_directory_name=FTP_DIRECTORY_NAME)
        self.assertTrue(this_file_name == JSON_FILE_NAME_ON_FTP)

    def test_find_raw_file_json(self):
        """Ensures correct output from find_raw_file.

        In this case the file type is JSON.
        """

        this_file_name = probsevere_io.find_raw_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            unix_time_sec=VALID_TIME_UNIX_SEC,
            file_extension=probsevere_io.JSON_FILE_EXTENSION,
            raise_error_if_missing=False)
        self.assertTrue(this_file_name == JSON_FILE_NAME)

    def test_find_raw_file_ascii(self):
        """Ensures correct output from find_raw_file.

        In this case the file type is ASCII.
        """

        this_file_name = probsevere_io.find_raw_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            unix_time_sec=VALID_TIME_UNIX_SEC,
            file_extension=probsevere_io.ASCII_FILE_EXTENSION,
            raise_error_if_missing=False)
        self.assertTrue(this_file_name == ASCII_FILE_NAME)


if __name__ == '__main__':
    unittest.main()
