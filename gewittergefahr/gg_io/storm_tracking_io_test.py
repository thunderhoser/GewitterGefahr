"""Unit tests for storm_tracking_io.py."""

import unittest
import numpy
import pandas
from gewittergefahr.gg_io import storm_tracking_io as tracking_io

FAKE_DATA_SOURCE = 'foo'
FAKE_BUFFER_COLUMN_NAME = 'bar'

MIN_DISTANCE_BUFFER_METRES = 0.
MAX_DISTANCE_BUFFER_METRES = 5000.
BUFFER_COLUMN_NAME_POLYGON_INCLUDED = 'polygon_object_latlng_buffer_5000m'
BUFFER_COLUMN_NAME_POLYGON_EXCLUDED = 'polygon_object_latlng_buffer_0m_5000m'

# The following constants are used to test _get_pathless_processed_file_name,
# _get_relative_processed_directory, and find_processed_file.
UNIX_TIME_SEC = 1507167848  # 014408 UTC 5 Oct 2017
SPC_DATE_STRING = '20171004'
TRACKING_SCALE_METRES2 = 5e7
PATHLESS_SEGMOTION_FILE_NAME = 'storm-tracking_segmotion_2017-10-05-014408.p'
PATHLESS_PROBSEVERE_FILE_NAME = 'storm-tracking_probSevere_2017-10-05-014408.p'

RELATIVE_SEGMOTION_DIR_NAME = '20171004/scale_50000000m2'
RELATIVE_PROBSEVERE_DIR_NAME = '20171005/scale_50000000m2'
TOP_PROCESSED_DIR_NAME_SEGMOTION = 'segmotion'
TOP_PROCESSED_DIR_NAME_PROBSEVERE = 'probSevere'
SEGMOTION_FILE_NAME = (
    'segmotion/20171004/scale_50000000m2/'
    'storm-tracking_segmotion_2017-10-05-014408.p')
PROBSEVERE_FILE_NAME = (
    'probSevere/20171005/scale_50000000m2/'
    'storm-tracking_probSevere_2017-10-05-014408.p')

# The following constants are used to test remove_rows_with_nan.
ARRAY_WITHOUT_NAN = numpy.array([1, 2, 3, 4, 5])
ARRAY_WITH_NAN = numpy.array([1, numpy.nan, 3, numpy.nan, 5])

THIS_DICTIONARY = {'without_nan': ARRAY_WITHOUT_NAN,
                   'with_nan': ARRAY_WITH_NAN}
DATAFRAME_WITH_NAN = pandas.DataFrame.from_dict(THIS_DICTIONARY)

ROWS_WITH_NAN = numpy.array([1, 3], dtype=int)
DATAFRAME_WITHOUT_NAN = DATAFRAME_WITH_NAN.drop(
    DATAFRAME_WITH_NAN.index[ROWS_WITH_NAN], axis=0, inplace=False)


class StormTrackingIoTests(unittest.TestCase):
    """Each method is a unit test for storm_tracking_io.py."""

    def test_check_data_source_fake(self):
        """Ensures correct output from _check_data_source.

        In this case, input is an unrecognized data source.
        """

        with self.assertRaises(ValueError):
            tracking_io._check_data_source(FAKE_DATA_SOURCE)

    def test_check_data_source_segmotion(self):
        """Ensures correct output from _check_data_source.

        In this case, input a recognized data source.
        """

        tracking_io._check_data_source(tracking_io.SEGMOTION_SOURCE_ID)

    def test_check_data_source_probsevere(self):
        """Ensures correct output from _check_data_source.

        In this case, input a recognized data source.
        """

        tracking_io._check_data_source(tracking_io.PROBSEVERE_SOURCE_ID)

    def test_column_name_to_distance_buffer_polygon_included(self):
        """Ensures correct output from column_name_to_distance_buffer.

        In this case the original polygon is included in the distance buffer.
        """

        this_min_distance_metres, this_max_distance_metres = (
            tracking_io.column_name_to_distance_buffer(
                BUFFER_COLUMN_NAME_POLYGON_INCLUDED))
        self.assertTrue(numpy.isnan(this_min_distance_metres))
        self.assertTrue(this_max_distance_metres == MAX_DISTANCE_BUFFER_METRES)

    def test_column_name_to_distance_buffer_polygon_excluded(self):
        """Ensures correct output from column_name_to_distance_buffer.

        In this case the original polygon is excluded from the distance buffer.
        """

        this_min_distance_metres, this_max_distance_metres = (
            tracking_io.column_name_to_distance_buffer(
                BUFFER_COLUMN_NAME_POLYGON_EXCLUDED))
        self.assertTrue(this_min_distance_metres == MIN_DISTANCE_BUFFER_METRES)
        self.assertTrue(this_max_distance_metres == MAX_DISTANCE_BUFFER_METRES)

    def test_column_name_to_distance_buffer_fake(self):
        """Ensures correct output from column_name_to_distance_buffer.

        In this case the column name is fake (does not correspond to a distance
        buffer).
        """

        this_min_distance_metres, this_max_distance_metres = (
            tracking_io.column_name_to_distance_buffer(
                FAKE_BUFFER_COLUMN_NAME))
        self.assertTrue(this_min_distance_metres is None)
        self.assertTrue(this_max_distance_metres is None)

    def test_get_pathless_processed_file_name_segmotion(self):
        """Ensures correct output from _get_pathless_processed_file_name.

        In this case, data source is segmotion.
        """

        this_pathless_file_name = tracking_io._get_pathless_processed_file_name(
            UNIX_TIME_SEC, tracking_io.SEGMOTION_SOURCE_ID)
        self.assertTrue(this_pathless_file_name == PATHLESS_SEGMOTION_FILE_NAME)

    def test_get_pathless_processed_file_name_probsevere(self):
        """Ensures correct output from _get_pathless_processed_file_name.

        In this case, data source is probSevere.
        """

        this_pathless_file_name = tracking_io._get_pathless_processed_file_name(
            UNIX_TIME_SEC, tracking_io.PROBSEVERE_SOURCE_ID)
        self.assertTrue(this_pathless_file_name ==
                        PATHLESS_PROBSEVERE_FILE_NAME)

    def test_get_relative_processed_directory_segmotion(self):
        """Ensures correct output from _get_relative_processed_directory.

        In this case, data source is segmotion.
        """

        this_relative_dir_name = tracking_io._get_relative_processed_directory(
            data_source=tracking_io.SEGMOTION_SOURCE_ID,
            spc_date_string=SPC_DATE_STRING,
            tracking_scale_metres2=TRACKING_SCALE_METRES2)
        self.assertTrue(this_relative_dir_name == RELATIVE_SEGMOTION_DIR_NAME)

    def test_get_relative_processed_directory_probsevere(self):
        """Ensures correct output from _get_relative_processed_directory.

        In this case, data source is probSevere.
        """

        this_relative_dir_name = tracking_io._get_relative_processed_directory(
            data_source=tracking_io.PROBSEVERE_SOURCE_ID,
            unix_time_sec=UNIX_TIME_SEC,
            tracking_scale_metres2=TRACKING_SCALE_METRES2)
        self.assertTrue(this_relative_dir_name == RELATIVE_PROBSEVERE_DIR_NAME)

    def test_remove_rows_with_nan(self):
        """Ensures correct output from remove_rows_with_nan."""

        this_dataframe = tracking_io.remove_rows_with_nan(DATAFRAME_WITH_NAN)
        self.assertTrue(this_dataframe.equals(DATAFRAME_WITHOUT_NAN))

    def test_distance_buffer_to_column_name_polygon_included(self):
        """Ensures correct output from distance_buffer_to_column_name.

        In this case the original polygon is included in the distance buffer.
        """

        this_column_name = tracking_io.distance_buffer_to_column_name(
            numpy.nan, MAX_DISTANCE_BUFFER_METRES)
        self.assertTrue(this_column_name == BUFFER_COLUMN_NAME_POLYGON_INCLUDED)

    def test_distance_buffer_to_column_name_polygon_excluded(self):
        """Ensures correct output from distance_buffer_to_column_name.

        In this case the original polygon is excluded from the distance buffer.
        """

        this_column_name = tracking_io.distance_buffer_to_column_name(
            MIN_DISTANCE_BUFFER_METRES, MAX_DISTANCE_BUFFER_METRES)
        self.assertTrue(this_column_name == BUFFER_COLUMN_NAME_POLYGON_EXCLUDED)

    def test_find_processed_file_segmotion(self):
        """Ensures correct output from find_processed_file.

        In this case, data source is segmotion.
        """

        this_processed_file_name = tracking_io.find_processed_file(
            unix_time_sec=UNIX_TIME_SEC,
            data_source=tracking_io.SEGMOTION_SOURCE_ID,
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
            data_source=tracking_io.PROBSEVERE_SOURCE_ID,
            top_processed_dir_name=TOP_PROCESSED_DIR_NAME_PROBSEVERE,
            tracking_scale_metres2=TRACKING_SCALE_METRES2,
            raise_error_if_missing=False)
        self.assertTrue(this_processed_file_name == PROBSEVERE_FILE_NAME)


if __name__ == '__main__':
    unittest.main()
