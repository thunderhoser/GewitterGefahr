"""Unit tests for probsevere_io.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_io import probsevere_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

TOP_DIRECTORY_NAME = 'foo'
FTP_DIRECTORY_NAME = 'bar'
VALID_TIME_UNIX_SEC = 1507181187  # 052627 5 Oct 2017

PATHLESS_JSON_FILE_NAME = 'SSEC_AWIPS_PROBSEVERE_20171005_052627.json'
PATHLESS_ASCII_FILE_NAME = 'SSEC_AWIPS_PROBSEVERE_20171005_052627.ascii'

JSON_FILE_NAME = (
    'foo/201710/20171005/SSEC_AWIPS_PROBSEVERE_20171005_052627.json'
)
ASCII_FILE_NAME = (
    'foo/201710/20171005/SSEC_AWIPS_PROBSEVERE_20171005_052627.ascii'
)
ALTERNATIVE_JSON_FILE_NAME = (
    'foo/201710/20171005/SSEC_AWIPS_CONVECTPROB_20171005_052627.json'
)
ALTERNATIVE_ASCII_FILE_NAME = (
    'foo/201710/20171005/SSEC_AWIPS_CONVECTPROB_20171005_052627.ascii'
)

JSON_FILE_NAME_ON_FTP = 'bar/SSEC_AWIPS_PROBSEVERE_20171005_052627.json'

# The following constants are used to test
# _get_dates_needed_for_renaming_storms.
NUM_DATES_IN_PERIOD = 20
WORKING_DATE_INDEX_MIDDLE = 10

DATE_NEEDED_INDICES_START = numpy.array([0, 1], dtype=int)
DATE_NEEDED_INDICES_END = numpy.array([18, 19], dtype=int)
DATE_NEEDED_INDICES_MIDDLE = numpy.array([9, 10, 11], dtype=int)

# The following constants are used to test _rename_storms_one_original_id.
STORM_TIMES_UNIX_SEC = numpy.array(
    [0, 1, 2, 3, 6, 7, 8, 9, 12, 15], dtype=int
)

NEXT_ID_NUMBER = 5
MAX_DROPOUT_TIME_SECONDS = 2

STORM_ID_STRINGS = [
    '5_probSevere', '5_probSevere', '5_probSevere', '5_probSevere',
    '6_probSevere', '6_probSevere', '6_probSevere', '6_probSevere',
    '7_probSevere', '8_probSevere'
]

NEXT_ID_NUMBER_AFTER_ONE_ORIG_ID = 9

# The following constants are used to test _rename_storms_one_table.
WORKING_DATE_INDEX_FOR_TABLE = 1

THESE_ID_STRINGS = [
    'a', 'b', 'c',
    'b', 'd',
    'a', 'b', 'd',
    'a', 'b', 'c'
]

THESE_TIMES_UNIX_SEC = numpy.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3], dtype=int)
THESE_DATE_INDICES = numpy.full(
    THESE_TIMES_UNIX_SEC.shape, WORKING_DATE_INDEX_FOR_TABLE, dtype=int
)

THIS_DICT = {
    tracking_utils.PRIMARY_ID_COLUMN: THESE_ID_STRINGS,
    tracking_utils.VALID_TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    probsevere_io.DATE_INDEX_KEY: THESE_DATE_INDICES
}
STORM_OBJECT_TABLE_ORIG_IDS_1DAY = pandas.DataFrame.from_dict(THIS_DICT)

THESE_ID_STRINGS = [
    '5_probSevere', '6_probSevere', '7_probSevere',
    '6_probSevere', '9_probSevere',
    '5_probSevere', '6_probSevere', '9_probSevere',
    '5_probSevere', '6_probSevere', '8_probSevere'
]

THIS_DICT = {
    tracking_utils.PRIMARY_ID_COLUMN: THESE_ID_STRINGS,
    tracking_utils.VALID_TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    probsevere_io.DATE_INDEX_KEY: THESE_DATE_INDICES
}

STORM_OBJECT_TABLE_NEW_IDS_1DAY = pandas.DataFrame.from_dict(THIS_DICT)
NEXT_ID_NUMBER_AFTER_1DAY = 10

THESE_ID_STRINGS = [
    'a', 'b', 'c',
    'a', 'b', 'c',
    'a', 'c', 'd'
]

THESE_TIMES_UNIX_SEC = numpy.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=int)
THESE_DATE_INDICES = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)

THIS_DICT = {
    tracking_utils.PRIMARY_ID_COLUMN: THESE_ID_STRINGS,
    tracking_utils.VALID_TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    probsevere_io.DATE_INDEX_KEY: THESE_DATE_INDICES
}
STORM_OBJECT_TABLE_ORIG_IDS_2DAYS = pandas.DataFrame.from_dict(THIS_DICT)

THESE_ID_STRINGS = [
    '5_probSevere', 'b', '6_probSevere',
    '5_probSevere', 'b', '6_probSevere',
    '5_probSevere', '6_probSevere', '7_probSevere'
]

THIS_DICT = {
    tracking_utils.PRIMARY_ID_COLUMN: THESE_ID_STRINGS,
    tracking_utils.VALID_TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    probsevere_io.DATE_INDEX_KEY: THESE_DATE_INDICES
}

STORM_OBJECT_TABLE_NEW_IDS_2DAYS = pandas.DataFrame.from_dict(THIS_DICT)
NEXT_ID_NUMBER_AFTER_2DAYS = 8


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

    def test_raw_file_name_to_time_json(self):
        """Ensures correct output from raw_file_name_to_time.

        In this case, the file type is JSON and file name does *not* include the
        alternative prefix.
        """

        this_time_unix_sec = probsevere_io.raw_file_name_to_time(JSON_FILE_NAME)
        self.assertTrue(this_time_unix_sec == VALID_TIME_UNIX_SEC)

    def test_raw_file_name_to_time_json_alternative(self):
        """Ensures correct output from raw_file_name_to_time.

        In this case, the file type is JSON and file name includes the
        alternative prefix.
        """

        this_time_unix_sec = probsevere_io.raw_file_name_to_time(
            ALTERNATIVE_JSON_FILE_NAME)

        self.assertTrue(this_time_unix_sec == VALID_TIME_UNIX_SEC)

    def test_raw_file_name_to_time_ascii(self):
        """Ensures correct output from raw_file_name_to_time.

        In this case, the file type is ASCII and file name does *not* include the
        alternative prefix.
        """

        this_time_unix_sec = probsevere_io.raw_file_name_to_time(
            ASCII_FILE_NAME)

        self.assertTrue(this_time_unix_sec == VALID_TIME_UNIX_SEC)

    def test_raw_file_name_to_time_ascii_alternative(self):
        """Ensures correct output from raw_file_name_to_time.

        In this case, the file type is ASCII and file name includes the
        alternative prefix.
        """

        this_time_unix_sec = probsevere_io.raw_file_name_to_time(
            ALTERNATIVE_ASCII_FILE_NAME)

        self.assertTrue(this_time_unix_sec == VALID_TIME_UNIX_SEC)

    def test_get_dates_needed_for_renaming_storms_start(self):
        """Ensures correct output from _get_dates_needed_for_renaming_storms.

        In this case, working date is the first.
        """

        these_indices = probsevere_io._get_dates_needed_for_renaming_storms(
            working_date_index=0, num_dates_in_period=NUM_DATES_IN_PERIOD)

        self.assertTrue(numpy.array_equal(
            these_indices, DATE_NEEDED_INDICES_START
        ))

    def test_get_dates_needed_for_renaming_storms_end(self):
        """Ensures correct output from _get_dates_needed_for_renaming_storms.

        In this case, working date is the last.
        """

        these_indices = probsevere_io._get_dates_needed_for_renaming_storms(
            working_date_index=NUM_DATES_IN_PERIOD - 1,
            num_dates_in_period=NUM_DATES_IN_PERIOD)

        self.assertTrue(numpy.array_equal(
            these_indices, DATE_NEEDED_INDICES_END
        ))

    def test_get_dates_needed_for_renaming_storms_middle(self):
        """Ensures correct output from _get_dates_needed_for_renaming_storms.

        In this case, working date is the middle one.
        """

        these_indices = probsevere_io._get_dates_needed_for_renaming_storms(
            working_date_index=WORKING_DATE_INDEX_MIDDLE,
            num_dates_in_period=NUM_DATES_IN_PERIOD)

        self.assertTrue(numpy.array_equal(
            these_indices, DATE_NEEDED_INDICES_MIDDLE
        ))

    def test_rename_storms_one_original_id(self):
        """Ensures correct output from _rename_storms_one_original_id."""

        these_id_strings, this_id_number = (
            probsevere_io._rename_storms_one_original_id(
                valid_times_unix_sec=STORM_TIMES_UNIX_SEC,
                next_id_number=NEXT_ID_NUMBER + 0,
                max_dropout_time_seconds=MAX_DROPOUT_TIME_SECONDS)
        )

        self.assertTrue(these_id_strings == STORM_ID_STRINGS)
        self.assertTrue(this_id_number == NEXT_ID_NUMBER_AFTER_ONE_ORIG_ID)

    def test_rename_storms_one_table_1day(self):
        """Ensures correct output from _rename_storms_one_table.

        In this case, the input table contains data from one day.
        """

        this_new_table, this_id_number = probsevere_io._rename_storms_one_table(
            storm_object_table=copy.deepcopy(STORM_OBJECT_TABLE_ORIG_IDS_1DAY),
            next_id_number=NEXT_ID_NUMBER,
            max_dropout_time_seconds=MAX_DROPOUT_TIME_SECONDS,
            working_date_index=WORKING_DATE_INDEX_FOR_TABLE)

        self.assertTrue(this_new_table.equals(STORM_OBJECT_TABLE_NEW_IDS_1DAY))
        self.assertTrue(this_id_number == NEXT_ID_NUMBER_AFTER_1DAY)

    def test_rename_storms_one_table_2days(self):
        """Ensures correct output from _rename_storms_one_table.

        In this case, the input table contains data from two days.
        """

        this_new_table, this_id_number = probsevere_io._rename_storms_one_table(
            storm_object_table=copy.deepcopy(STORM_OBJECT_TABLE_ORIG_IDS_2DAYS),
            next_id_number=NEXT_ID_NUMBER,
            max_dropout_time_seconds=MAX_DROPOUT_TIME_SECONDS,
            working_date_index=WORKING_DATE_INDEX_FOR_TABLE)

        self.assertTrue(this_new_table.equals(
            STORM_OBJECT_TABLE_NEW_IDS_2DAYS
        ))
        self.assertTrue(this_id_number == NEXT_ID_NUMBER_AFTER_2DAYS)


if __name__ == '__main__':
    unittest.main()
