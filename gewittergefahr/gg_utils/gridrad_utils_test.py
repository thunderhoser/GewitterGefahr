"""Unit tests for gridrad_utils.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import gridrad_utils

TOLERANCE = 1e-6

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

# These constants are used to test _get_field_name_for_echo_tops.
CRIT_REFL_0DECIMALS_DBZ = 40.
CRIT_REFL_1DECIMAL_DBZ = 40.1
CRIT_REFL_2DECIMALS_DBZ = 40.12

FIELD_NAME_0DECIMALS_NON_MYRORSS = 'echo_top_40.0dbz_km'
FIELD_NAME_1DECIMAL_NON_MYRORSS = 'echo_top_40.1dbz_km'
FIELD_NAME_2DECIMALS_NON_MYRORSS = 'echo_top_40.1dbz_km'

FIELD_NAME_0DECIMALS_MYRORSS = 'EchoTop_40.0'
FIELD_NAME_1DECIMAL_MYRORSS = 'EchoTop_40.1'
FIELD_NAME_2DECIMALS_MYRORSS = 'EchoTop_40.1'


class GridradUtilsTests(unittest.TestCase):
    """Each method is a unit test for gridrad_utils.py."""

    def test_get_field_name_for_echo_tops_0decimals_non_myrorss(self):
        """Ensures correct output from _get_field_name_for_echo_tops.

        In this case, critical reflectivity has 0 decimal places and field name
        will be in GewitterGefahr format.
        """

        this_field_name = gridrad_utils._get_field_name_for_echo_tops(
            CRIT_REFL_0DECIMALS_DBZ, False)
        self.assertTrue(this_field_name == FIELD_NAME_0DECIMALS_NON_MYRORSS)

    def test_get_field_name_for_echo_tops_0decimals_myrorss(self):
        """Ensures correct output from _get_field_name_for_echo_tops.

        In this case, critical reflectivity has 0 decimal places and field name
        will be in MYRORSS format.
        """

        this_field_name = gridrad_utils._get_field_name_for_echo_tops(
            CRIT_REFL_0DECIMALS_DBZ, True)
        self.assertTrue(this_field_name == FIELD_NAME_0DECIMALS_MYRORSS)

    def test_get_field_name_for_echo_tops_1decimal_non_myrorss(self):
        """Ensures correct output from _get_field_name_for_echo_tops.

        In this case, critical reflectivity has 1 decimal place and field name
        will be in GewitterGefahr format.
        """

        this_field_name = gridrad_utils._get_field_name_for_echo_tops(
            CRIT_REFL_1DECIMAL_DBZ, False)
        self.assertTrue(this_field_name == FIELD_NAME_1DECIMAL_NON_MYRORSS)

    def test_get_field_name_for_echo_tops_1decimal_myrorss(self):
        """Ensures correct output from _get_field_name_for_echo_tops.

        In this case, critical reflectivity has 1 decimal place and field name
        will be in MYRORSS format.
        """

        this_field_name = gridrad_utils._get_field_name_for_echo_tops(
            CRIT_REFL_1DECIMAL_DBZ, True)
        self.assertTrue(this_field_name == FIELD_NAME_1DECIMAL_MYRORSS)

    def test_get_field_name_for_echo_tops_2decimals_non_myrorss(self):
        """Ensures correct output from _get_field_name_for_echo_tops.

        In this case, critical reflectivity has 2 decimal places and field name
        will be in GewitterGefahr format.
        """

        this_field_name = gridrad_utils._get_field_name_for_echo_tops(
            CRIT_REFL_2DECIMALS_DBZ, False)
        self.assertTrue(this_field_name == FIELD_NAME_2DECIMALS_NON_MYRORSS)

    def test_get_field_name_for_echo_tops_2decimals_myrorss(self):
        """Ensures correct output from _get_field_name_for_echo_tops.

        In this case, critical reflectivity has 2 decimal places and field name
        will be in MYRORSS format.
        """

        this_field_name = gridrad_utils._get_field_name_for_echo_tops(
            CRIT_REFL_2DECIMALS_DBZ, True)
        self.assertTrue(this_field_name == FIELD_NAME_2DECIMALS_MYRORSS)

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
