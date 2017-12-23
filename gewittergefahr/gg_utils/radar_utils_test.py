"""Unit tests for radar_utils.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import radar_utils

TOLERANCE = 1e-6

# These constants are used to test get_echo_top.
UNIQUE_GRID_POINT_HEIGHTS_M_ASL = numpy.array([1000., 2000., 3000.])

THIS_REFL_MATRIX_1KM_DBZ = numpy.array(
    [[0., 10.], [20., 0., ], [0., numpy.nan]])
THIS_REFL_MATRIX_2KM_DBZ = numpy.array(
    [[20., 30.], [40., 10., ], [50., 50.]])
THIS_REFL_MATRIX_3KM_DBZ = numpy.array(
    [[40., 50.], [60., 20., ], [20., 20.]])
REFLECTIVITY_MATRIX_DBZ = numpy.stack(
    (THIS_REFL_MATRIX_1KM_DBZ, THIS_REFL_MATRIX_2KM_DBZ,
     THIS_REFL_MATRIX_3KM_DBZ), axis=0)

CRIT_REFL_FOR_ECHO_TOPS_DBZ = 40.
ECHO_TOP_MATRIX_M_ASL = numpy.array(
    [[3000., 3200.], [3333.333333, numpy.nan], [2333.333333, 2333.333333]])


class RadarUtilsTests(unittest.TestCase):
    """Each method is a unit test for radar_utils.py."""

    def test_get_echo_top_single_column(self):
        """Ensures correct output from get_echo_top_single_column."""

        this_num_rows = THIS_REFL_MATRIX_1KM_DBZ.shape[0]
        this_num_columns = THIS_REFL_MATRIX_1KM_DBZ.shape[1]
        this_echo_top_matrix_m_asl = numpy.full(
            (this_num_rows, this_num_columns), numpy.nan)

        for i in range(this_num_rows):
            for j in range(this_num_columns):
                this_echo_top_matrix_m_asl[i, j] = (
                    radar_utils.get_echo_top_single_column(
                        reflectivities_dbz=REFLECTIVITY_MATRIX_DBZ[:, i, j],
                        heights_m_asl=UNIQUE_GRID_POINT_HEIGHTS_M_ASL,
                        critical_reflectivity_dbz=CRIT_REFL_FOR_ECHO_TOPS_DBZ,
                        check_args=True))

        self.assertTrue(numpy.allclose(
            this_echo_top_matrix_m_asl, ECHO_TOP_MATRIX_M_ASL, atol=TOLERANCE,
            equal_nan=True))


if __name__ == '__main__':
    unittest.main()
