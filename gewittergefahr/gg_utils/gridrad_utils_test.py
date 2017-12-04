"""Unit tests for gridrad_utils.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import gridrad_utils

TOLERANCE = 1e-6

REFLECTIVITY_MATRIX_0KM_DBZ = numpy.array([[1., 2.], [3., 4.], [5., 6.]])
REFLECTIVITY_MATRIX_1KM_DBZ = numpy.array([[7., 8.], [9., 10.], [11., 12.]])
REFLECTIVITY_MATRIX_2KM_DBZ = numpy.array([[13., 14.], [15., 16.], [17., 18.]])
REFLECTIVITY_MATRIX_DBZ = numpy.stack(
    (REFLECTIVITY_MATRIX_0KM_DBZ, REFLECTIVITY_MATRIX_1KM_DBZ,
     REFLECTIVITY_MATRIX_2KM_DBZ), axis=0)

UNIQUE_GRID_POINT_HEIGHTS_M_ASL = numpy.array([0., 1000., 2000.])
TARGET_HEIGHT_MATRIX_M_ASL = numpy.array(
    [[-500., 0.], [500., 1500.], [2000., 2500]])
INTERP_REFL_MATRIX_DBZ = numpy.array([[-2., 2.], [6., 13.], [17., 21.]])


class GridradUtilsTests(unittest.TestCase):
    """Each method is a unit test for gridrad_utils.py."""

    def test_interp_reflectivity_to_heights(self):
        """Ensures correct output from interp_reflectivity_to_heights."""

        this_interp_matrix_dbz = gridrad_utils.interp_reflectivity_to_heights(
            reflectivity_matrix_dbz=REFLECTIVITY_MATRIX_DBZ,
            unique_grid_point_heights_m_asl=UNIQUE_GRID_POINT_HEIGHTS_M_ASL,
            target_height_matrix_m_asl=TARGET_HEIGHT_MATRIX_M_ASL)

        self.assertTrue(numpy.allclose(
            this_interp_matrix_dbz, INTERP_REFL_MATRIX_DBZ, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
