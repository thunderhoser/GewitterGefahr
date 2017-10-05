"""Unit tests for radar_sparse_to_full.py."""

import unittest
import numpy
import pandas
from gewittergefahr.gg_io import radar_io
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f

TOLERANCE = 1e-6

START_ROWS = numpy.array([0, 1, 2, 3], dtype=int)
START_COLUMNS = numpy.array([3, 2, 1, 0], dtype=int)
GRID_CELL_COUNTS = numpy.array([1, 2, 3, 4], dtype=int)
RADAR_VALUES = numpy.array([35., 50., 70., 65.])

NUM_GRID_ROWS = 4
NUM_GRID_COLUMNS = 6
RADAR_FIELD_NAME = radar_io.REFL_COLUMN_MAX_NAME

SPARSE_GRID_DICT = {radar_io.GRID_ROW_COLUMN: START_ROWS,
                    radar_io.GRID_COLUMN_COLUMN: START_COLUMNS,
                    radar_io.NUM_GRID_CELL_COLUMN: GRID_CELL_COUNTS,
                    RADAR_FIELD_NAME: RADAR_VALUES}
SPARSE_GRID_TABLE = pandas.DataFrame.from_dict(SPARSE_GRID_DICT)

FULL_MATRIX = numpy.array(
    [[numpy.nan, numpy.nan, numpy.nan, 35., numpy.nan, numpy.nan],
     [numpy.nan, numpy.nan, 50., 50., numpy.nan, numpy.nan],
     [numpy.nan, 70., 70., 70., numpy.nan, numpy.nan],
     [65., 65., 65., 65., numpy.nan, numpy.nan]])


class RadarSparseToFullTests(unittest.TestCase):
    """Each method is a unit test for radar_sparse_to_full.py."""

    def test_convert(self):
        """Ensures correct output from _convert."""

        this_full_matrix = radar_s2f._convert(
            SPARSE_GRID_TABLE, field_name=RADAR_FIELD_NAME,
            num_grid_rows=NUM_GRID_ROWS, num_grid_columns=NUM_GRID_COLUMNS)

        self.assertTrue(numpy.allclose(
            this_full_matrix, FULL_MATRIX, atol=TOLERANCE, equal_nan=True))


if __name__ == '__main__':
    unittest.main()
