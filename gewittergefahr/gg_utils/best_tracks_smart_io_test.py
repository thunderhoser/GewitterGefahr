"""Unit tests for best_tracks_smart_io.py."""

import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import best_tracks
from gewittergefahr.gg_utils import best_tracks_smart_io

THESE_STORM_IDS = [
    'foo', 'bar', 'hal',
    'foo', 'bar', 'moo', best_tracks.EMPTY_STORM_ID,
    'foo', 'moo', best_tracks.EMPTY_STORM_ID]
THESE_SPC_DATES_UNIX_SEC = numpy.array(
    [-1, -1, 0, 0, 0, 1, 1, 1, 2, 2], dtype=int)

THIS_DICT = {tracking_utils.STORM_ID_COLUMN: THESE_STORM_IDS,
             tracking_utils.SPC_DATE_COLUMN: THESE_SPC_DATES_UNIX_SEC}
STORM_OBJECT_TABLE = pandas.DataFrame.from_dict(THIS_DICT)

THESE_STORM_IDS = ['bar', 'foo', 'hal', 'moo']
THIS_DICT = {tracking_utils.STORM_ID_COLUMN: THESE_STORM_IDS}
STORM_TRACK_TABLE = pandas.DataFrame.from_dict(THIS_DICT)

THIS_NESTED_ARRAY = STORM_TRACK_TABLE[[
    tracking_utils.STORM_ID_COLUMN,
    tracking_utils.STORM_ID_COLUMN]].values.tolist()
THIS_DICT = {best_tracks.OBJECT_INDICES_COLUMN_FOR_TRACK: THIS_NESTED_ARRAY}
STORM_TRACK_TABLE = STORM_TRACK_TABLE.assign(**THIS_DICT)

STORM_TRACK_TABLE[best_tracks.OBJECT_INDICES_COLUMN_FOR_TRACK].values[
    0] = numpy.array([1, 4], dtype=int)
STORM_TRACK_TABLE[best_tracks.OBJECT_INDICES_COLUMN_FOR_TRACK].values[
    1] = numpy.array([0, 3, 7], dtype=int)
STORM_TRACK_TABLE[best_tracks.OBJECT_INDICES_COLUMN_FOR_TRACK].values[
    2] = numpy.array([2], dtype=int)
STORM_TRACK_TABLE[best_tracks.OBJECT_INDICES_COLUMN_FOR_TRACK].values[
    3] = numpy.array([5, 8], dtype=int)

TRACK_INDICES_FOR_SPC_DATE_MINUS1 = numpy.array([0, 1], dtype=int)
TRACK_INDICES_FOR_SPC_DATE0 = numpy.array([0, 1, 2], dtype=int)
TRACK_INDICES_FOR_SPC_DATE1 = numpy.array([1, 3], dtype=int)
TRACK_INDICES_FOR_SPC_DATE2 = numpy.array([3], dtype=int)


class BestTracksSmartIoTests(unittest.TestCase):
    """Each method is a unit test for best_tracks_smart_io.py."""

    def test_find_tracks_with_spc_date_minus1(self):
        """Ensures correct output from _find_tracks_with_spc_date.

        In this case, SPC date is -1.
        """

        these_track_indices = best_tracks_smart_io._find_tracks_with_spc_date(
            STORM_OBJECT_TABLE, STORM_TRACK_TABLE, spc_date_unix_sec=-1)
        self.assertTrue(numpy.array_equal(
            these_track_indices, TRACK_INDICES_FOR_SPC_DATE_MINUS1))

    def test_find_tracks_with_spc_date0(self):
        """Ensures correct output from _find_tracks_with_spc_date.

        In this case, SPC date is 0.
        """

        these_track_indices = best_tracks_smart_io._find_tracks_with_spc_date(
            STORM_OBJECT_TABLE, STORM_TRACK_TABLE, spc_date_unix_sec=0)
        self.assertTrue(numpy.array_equal(
            these_track_indices, TRACK_INDICES_FOR_SPC_DATE0))

    def test_find_tracks_with_spc_date1(self):
        """Ensures correct output from _find_tracks_with_spc_date.

        In this case, SPC date is 1.
        """

        these_track_indices = best_tracks_smart_io._find_tracks_with_spc_date(
            STORM_OBJECT_TABLE, STORM_TRACK_TABLE, spc_date_unix_sec=1)
        self.assertTrue(numpy.array_equal(
            these_track_indices, TRACK_INDICES_FOR_SPC_DATE1))

    def test_find_tracks_with_spc_date2(self):
        """Ensures correct output from _find_tracks_with_spc_date.

        In this case, SPC date is 2.
        """

        these_track_indices = best_tracks_smart_io._find_tracks_with_spc_date(
            STORM_OBJECT_TABLE, STORM_TRACK_TABLE, spc_date_unix_sec=2)
        self.assertTrue(numpy.array_equal(
            these_track_indices, TRACK_INDICES_FOR_SPC_DATE2))


if __name__ == '__main__':
    unittest.main()
