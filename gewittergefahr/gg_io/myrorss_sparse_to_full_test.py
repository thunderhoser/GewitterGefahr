"""Unit tests for myrorss_sparse_to_full.py."""

import unittest
import numpy
import pandas
from gewittergefahr.gg_io import myrorss_sparse_to_full as s2f
from gewittergefahr.gg_io import myrorss_io

TOLERANCE = 1e-6

START_ROWS = numpy.array([0, 1, 2, 3], dtype=int)
START_COLUMNS = numpy.array([3, 2, 1, 0], dtype=int)
GRID_CELL_COUNTS = numpy.array([1, 2, 3, 4], dtype=int)
RADAR_VALUES = numpy.array([35., 50., 70., 65.])

NUM_LAT_IN_GRID = 4
NUM_LNG_IN_GRID = 6
RADAR_FIELD_NAME = 'reflectivity_column_max_dbz'

SPARSE_GRID_DICT = {myrorss_io.GRID_ROW_COLUMN: START_ROWS,
                    myrorss_io.GRID_COLUMN_COLUMN: START_COLUMNS,
                    myrorss_io.NUM_GRID_CELL_COLUMN: GRID_CELL_COUNTS,
                    RADAR_FIELD_NAME: RADAR_VALUES}
SPARSE_GRID_TABLE = pandas.DataFrame.from_dict(SPARSE_GRID_DICT)

EXPECTED_FULL_MATRIX = numpy.array(
    [[numpy.nan, numpy.nan, numpy.nan, 35., numpy.nan, numpy.nan],
     [numpy.nan, numpy.nan, 50., 50., numpy.nan, numpy.nan],
     [numpy.nan, 70., 70., 70., numpy.nan, numpy.nan],
     [65., 65., 65., 65., numpy.nan, numpy.nan]])


class MyrorssSparseToFullTests(unittest.TestCase):
    """Each method is a unit test for myrorss_sparse_to_full.py."""

    def test_sparse_to_full_grid(self):
        """Ensures correct output from sparse_to_full_grid."""

        this_full_matrix = s2f.sparse_to_full_grid(
            SPARSE_GRID_TABLE, field_name=RADAR_FIELD_NAME,
            num_lat_in_grid=NUM_LAT_IN_GRID, num_lng_in_grid=NUM_LNG_IN_GRID)

        self.assertTrue(numpy.allclose(
            this_full_matrix, EXPECTED_FULL_MATRIX, atol=TOLERANCE,
            equal_nan=True))


if __name__ == '__main__':
    unittest.main()
