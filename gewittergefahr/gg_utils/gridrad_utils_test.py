"""Unit tests for gridrad_utils.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import gridrad_utils
from gewittergefahr.gg_utils import radar_utils

TOLERANCE = 1e-6

# The following constants are used to test fields_and_refl_heights_to_pairs.
FIELD_NAMES = [
    radar_utils.REFL_NAME, radar_utils.DIFFERENTIAL_REFL_NAME,
    radar_utils.SPEC_DIFF_PHASE_NAME, radar_utils.CORRELATION_COEFF_NAME,
    radar_utils.SPECTRUM_WIDTH_NAME, radar_utils.VORTICITY_NAME,
    radar_utils.DIVERGENCE_NAME]
HEIGHTS_M_ASL = numpy.array([500, 2500, 5000, 10000])

FIELD_NAME_BY_PAIR = (
    [radar_utils.REFL_NAME] * len(HEIGHTS_M_ASL) +
    [radar_utils.DIFFERENTIAL_REFL_NAME] * len(HEIGHTS_M_ASL) +
    [radar_utils.SPEC_DIFF_PHASE_NAME] * len(HEIGHTS_M_ASL) +
    [radar_utils.CORRELATION_COEFF_NAME] * len(HEIGHTS_M_ASL) +
    [radar_utils.SPECTRUM_WIDTH_NAME] * len(HEIGHTS_M_ASL) +
    [radar_utils.VORTICITY_NAME] * len(HEIGHTS_M_ASL) +
    [radar_utils.DIVERGENCE_NAME] * len(HEIGHTS_M_ASL))

HEIGHT_BY_PAIR_M_ASL = numpy.array([500, 2500, 5000, 10000,
                                    500, 2500, 5000, 10000,
                                    500, 2500, 5000, 10000,
                                    500, 2500, 5000, 10000,
                                    500, 2500, 5000, 10000,
                                    500, 2500, 5000, 10000,
                                    500, 2500, 5000, 10000])

# These constants are used to test interp_reflectivity_to_heights.
THIS_REFL_MATRIX_1KM_DBZ = numpy.array(
    [[1., numpy.nan], [3., numpy.nan], [5., numpy.nan]])
THIS_REFL_MATRIX_2KM_DBZ = numpy.array(
    [[7., 8.], [9., numpy.nan], [11., numpy.nan]])
THIS_REFL_MATRIX_3KM_DBZ = numpy.array(
    [[13., 14.], [15., 16.], [17., numpy.nan]])
REFLECTIVITY_MATRIX_DBZ = numpy.stack(
    (THIS_REFL_MATRIX_1KM_DBZ, THIS_REFL_MATRIX_2KM_DBZ,
     THIS_REFL_MATRIX_3KM_DBZ), axis=0)

UNIQUE_GRID_POINT_HEIGHTS_M_ASL = numpy.array([1000., 2000., 3000.])
TARGET_HEIGHT_MATRIX_M_ASL = numpy.array(
    [[500., 1000.], [1500., 2500.], [3000., 3500.]])
INTERP_REFL_MATRIX_DBZ = numpy.array(
    [[-2., 2.], [6., numpy.nan], [17., numpy.nan]])

# These constants are used to test get_column_max_reflectivity.
COLUMN_MAX_REFL_MATRIX_DBZ = numpy.array(
    [[13., 14.], [15., 16.], [17., numpy.nan]])

# These constants are used to test get_echo_top.
THIS_REFL_MATRIX_1KM_DBZ = numpy.array(
    [[0., 10.], [20., 0.,], [0., numpy.nan]])
THIS_REFL_MATRIX_2KM_DBZ = numpy.array(
    [[20., 30.], [40., 10.,], [50., 50.]])
THIS_REFL_MATRIX_3KM_DBZ = numpy.array(
    [[40., 50.], [60., 20.,], [20., 20.]])
REFL_MATRIX_FOR_ECHO_TOPS_DBZ = numpy.stack(
    (THIS_REFL_MATRIX_1KM_DBZ, THIS_REFL_MATRIX_2KM_DBZ,
     THIS_REFL_MATRIX_3KM_DBZ), axis=0)

CRIT_REFL_FOR_ECHO_TOPS_DBZ = 40.
ECHO_TOP_MATRIX_M_ASL = numpy.array(
    [[3000., 3200.], [3333.333333, numpy.nan], [2333.333333, 2333.333333]])


class GridradUtilsTests(unittest.TestCase):
    """Each method is a unit test for gridrad_utils.py."""

    def test_fields_and_refl_heights_to_pairs(self):
        """Ensures correct output from fields_and_refl_heights_to_pairs."""

        this_field_name_by_pair, this_height_by_pair_m_asl = (
            gridrad_utils.fields_and_refl_heights_to_pairs(
                field_names=FIELD_NAMES, heights_m_asl=HEIGHTS_M_ASL))

        self.assertTrue(this_field_name_by_pair == FIELD_NAME_BY_PAIR)
        self.assertTrue(numpy.array_equal(
            this_height_by_pair_m_asl, HEIGHT_BY_PAIR_M_ASL))

    def test_interp_reflectivity_to_heights(self):
        """Ensures correct output from interp_reflectivity_to_heights."""

        this_interp_matrix_dbz = gridrad_utils.interp_reflectivity_to_heights(
            reflectivity_matrix_dbz=REFLECTIVITY_MATRIX_DBZ,
            unique_grid_point_heights_m_asl=UNIQUE_GRID_POINT_HEIGHTS_M_ASL,
            target_height_matrix_m_asl=TARGET_HEIGHT_MATRIX_M_ASL)

        self.assertTrue(numpy.allclose(
            this_interp_matrix_dbz, INTERP_REFL_MATRIX_DBZ, atol=TOLERANCE,
            equal_nan=True))

    def test_get_column_max_reflectivity(self):
        """Ensures correct output from get_column_max_reflectivity."""

        this_column_max_matrix_dbz = gridrad_utils.get_column_max_reflectivity(
            REFLECTIVITY_MATRIX_DBZ)
        self.assertTrue(numpy.allclose(
            this_column_max_matrix_dbz, COLUMN_MAX_REFL_MATRIX_DBZ,
            atol=TOLERANCE, equal_nan=True))

    def test_get_echo_tops(self):
        """Ensures correct output from get_echo_tops."""

        this_echo_top_matrix_m_asl = gridrad_utils.get_echo_tops(
            reflectivity_matrix_dbz=REFL_MATRIX_FOR_ECHO_TOPS_DBZ,
            unique_grid_point_heights_m_asl=UNIQUE_GRID_POINT_HEIGHTS_M_ASL,
            critical_reflectivity_dbz=CRIT_REFL_FOR_ECHO_TOPS_DBZ)

        self.assertTrue(numpy.allclose(
            this_echo_top_matrix_m_asl, ECHO_TOP_MATRIX_M_ASL, atol=TOLERANCE,
            equal_nan=True))


if __name__ == '__main__':
    unittest.main()
