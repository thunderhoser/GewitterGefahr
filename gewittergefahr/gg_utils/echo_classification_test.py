"""Unit tests for echo_classification.py."""

import copy
import unittest
import numpy
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import echo_classification as echo_classifn

TOLERANCE = 1e-6

# The following constants are used to test _estimate_melting_levels.
MELTING_LEVEL_LATITUDES_DEG = numpy.linspace(-90., 90., num=19)
MELTING_LEVEL_TIME_UNIX_SEC = 1541823287  # 041447 UTC 10 Nov 2018
MELTING_LEVELS_M_ASL = (
    echo_classifn.MELT_LEVEL_INTERCEPT_BY_MONTH_M_ASL[10] +
    echo_classifn.MELT_LEVEL_SLOPE_BY_MONTH_M_DEG01[10] *
    numpy.absolute(MELTING_LEVEL_LATITUDES_DEG)
)

# The following constants are used to test _neigh_metres_to_rowcol.
LARGE_GRID_HEIGHTS_M_ASL = radar_utils.get_valid_heights(
    data_source=radar_utils.MYRORSS_SOURCE_ID, field_name=radar_utils.REFL_NAME)

LARGE_GRID_METADATA_DICT = {
    echo_classifn.MIN_LATITUDE_KEY: 20.,
    echo_classifn.LATITUDE_SPACING_KEY: 0.01,
    echo_classifn.MIN_LONGITUDE_KEY: 230.,
    echo_classifn.LONGITUDE_SPACING_KEY: 0.01,
    echo_classifn.HEIGHTS_KEY: LARGE_GRID_HEIGHTS_M_ASL
}

THESE_LATITUDES_DEG, THESE_LONGITUDES_DEG = grids.get_latlng_grid_points(
    min_latitude_deg=LARGE_GRID_METADATA_DICT[echo_classifn.MIN_LATITUDE_KEY],
    min_longitude_deg=LARGE_GRID_METADATA_DICT[echo_classifn.MIN_LONGITUDE_KEY],
    lat_spacing_deg=LARGE_GRID_METADATA_DICT[
        echo_classifn.LATITUDE_SPACING_KEY],
    lng_spacing_deg=LARGE_GRID_METADATA_DICT[
        echo_classifn.LONGITUDE_SPACING_KEY],
    num_rows=7001, num_columns=3501)

LARGE_GRID_METADATA_DICT[echo_classifn.LATITUDES_KEY] = THESE_LATITUDES_DEG
LARGE_GRID_METADATA_DICT[echo_classifn.LONGITUDES_KEY] = THESE_LONGITUDES_DEG

LARGE_RADIUS_METRES = 12000.
NUM_ROWS_IN_LARGE_NEIGH = 23
NUM_COLUMNS_IN_LARGE_NEIGH = 29

GRID_METADATA_DICT = {
    echo_classifn.MIN_LATITUDE_KEY: 35.1,
    echo_classifn.LATITUDE_SPACING_KEY: 0.2,
    echo_classifn.MIN_LONGITUDE_KEY: 262.1,
    echo_classifn.LONGITUDE_SPACING_KEY: 0.2,
    echo_classifn.HEIGHTS_KEY: numpy.array([1000, 4000, 7000])
}

THESE_LATITUDES_DEG, THESE_LONGITUDES_DEG = grids.get_latlng_grid_points(
    min_latitude_deg=GRID_METADATA_DICT[echo_classifn.MIN_LATITUDE_KEY],
    min_longitude_deg=GRID_METADATA_DICT[echo_classifn.MIN_LONGITUDE_KEY],
    lat_spacing_deg=GRID_METADATA_DICT[echo_classifn.LATITUDE_SPACING_KEY],
    lng_spacing_deg=GRID_METADATA_DICT[echo_classifn.LONGITUDE_SPACING_KEY],
    num_rows=5, num_columns=7)

GRID_METADATA_DICT[echo_classifn.LATITUDES_KEY] = THESE_LATITUDES_DEG
GRID_METADATA_DICT[echo_classifn.LONGITUDES_KEY] = THESE_LONGITUDES_DEG

NEIGH_RADIUS_METRES = 12000.
NUM_ROWS_IN_NEIGH = 3
NUM_COLUMNS_IN_NEIGH = 3

# The following constants are used to test _get_peakedness.
THIS_FIRST_MATRIX = numpy.array([[0, 1, 2, 3, 4, 5, 6],
                                 [0, 1, 2, 3, 4, 5, 6],
                                 [0, 1, 2, 3, 4, 5, 6],
                                 [0, 1, 2, 3, 4, 5, 6],
                                 [0, 1, 2, 3, 4, 5, 20]])

THIS_SECOND_MATRIX = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                                  [2, 2, 2, 2, 2, 2, 2],
                                  [4, 4, 4, 4, 4, 4, 4],
                                  [6, 6, 6, 6, 6, 6, 6],
                                  [8, 8, 8, 8, 8, 8, 20]])

THIS_THIRD_MATRIX = numpy.array([[0, 1, 2, 3, 4, 5, 6],
                                 [3, 4, 5, 6, 7, 8, 9],
                                 [6, 7, 8, 9, 10, 11, 12],
                                 [9, 10, 11, 12, 13, 14, 15],
                                 [12, 13, 14, 15, 16, 17, 20]])

REFLECTIVITY_MATRIX_DBZ = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX, THIS_THIRD_MATRIX), axis=-1
).astype(float)

THIS_FIRST_MATRIX = numpy.array([[0, 0, 1, 2, 3, 4, 0],
                                 [0, 1, 2, 3, 4, 5, 5],
                                 [0, 1, 2, 3, 4, 5, 5],
                                 [0, 1, 2, 3, 4, 5, 5],
                                 [0, 0, 1, 2, 3, 4, 0]])

THIS_SECOND_MATRIX = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                                  [0, 2, 2, 2, 2, 2, 0],
                                  [2, 4, 4, 4, 4, 4, 2],
                                  [4, 6, 6, 6, 6, 6, 4],
                                  [0, 6, 6, 6, 6, 6, 0]])

THIS_THIRD_MATRIX = numpy.array([[0, 1, 2, 3, 4, 5, 0],
                                 [1, 4, 5, 6, 7, 8, 6],
                                 [4, 7, 8, 9, 10, 11, 9],
                                 [7, 10, 11, 12, 13, 14, 12],
                                 [0, 10, 11, 12, 13, 14, 0]])

THIS_MATRIX = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX, THIS_THIRD_MATRIX), axis=-1
).astype(float)
PEAKEDNESS_MATRIX_DBZ = REFLECTIVITY_MATRIX_DBZ - THIS_MATRIX

# The following constants are used to test _apply_convective_criterion1.
MAX_PEAKEDNESS_HEIGHT_M_ASL = 9000.

CRITERION1_FLAG_MATRIX = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 0, 0, 1]], dtype=bool)

# The following constants are used to test _apply_convective_criterion2.
VALID_TIME_UNIX_SEC = 1541823287  # 041447 UTC 10 Nov 2018
MIN_COMPOSITE_REFL_AML_DBZ = 15.

CRITERION2_FLAG_MATRIX = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 1],
                                      [1, 0, 0, 1, 1, 1, 1]], dtype=bool)

# The following constants are used to test _apply_convective_criterion3.
MIN_ECHO_TOP_M_ASL = 6000.
ECHO_TOP_LEVEL_DBZ = 13.

CRITERION3_FLAG_MATRIX = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 1, 1, 1],
                                      [1, 1, 1, 1, 1, 1, 1]], dtype=bool)

# The following constants are used to test _apply_convective_criterion4.
CRITERION4_FLAG_MATRIX = copy.deepcopy(CRITERION3_FLAG_MATRIX)

DUMMY_CRITERION3_FLAG_MATRIX = numpy.array([[1, 0, 0, 1, 1, 0, 1],
                                            [0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 1, 0, 0, 0, 1],
                                            [1, 0, 0, 1, 0, 1, 0],
                                            [1, 0, 0, 0, 1, 0, 0]], dtype=bool)

DUMMY_CRITERION4_FLAG_MATRIX = numpy.array([[0, 0, 0, 1, 1, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 1, 0, 0, 0, 1],
                                            [1, 0, 0, 1, 0, 1, 0],
                                            [1, 0, 0, 0, 1, 0, 0]], dtype=bool)

# The following constants are used to test _apply_convective_criterion5.
MIN_COMPOSITE_REFL_DBZ = 6.

CRITERION5_FLAG_MATRIX = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1, 1, 1]], dtype=bool)


class EchoClassificationTests(unittest.TestCase):
    """Each method is a unit test for echo_classification.py."""

    def test_estimate_melting_levels(self):
        """Ensures correct output from _estimate_melting_levels."""

        these_heights_m_asl = echo_classifn._estimate_melting_levels(
            latitudes_deg=MELTING_LEVEL_LATITUDES_DEG,
            valid_time_unix_sec=MELTING_LEVEL_TIME_UNIX_SEC)

        self.assertTrue(numpy.allclose(
            these_heights_m_asl, MELTING_LEVELS_M_ASL, atol=TOLERANCE))

    def test_neigh_metres_to_rowcol_large(self):
        """Ensures correct output from _neigh_metres_to_rowcol.

        In this case the grid is very large (3501 x 7001).
        """

        this_num_rows, this_num_columns = echo_classifn._neigh_metres_to_rowcol(
            neigh_radius_metres=LARGE_RADIUS_METRES,
            grid_metadata_dict=LARGE_GRID_METADATA_DICT)

        self.assertTrue(this_num_rows == NUM_ROWS_IN_LARGE_NEIGH)
        self.assertTrue(this_num_columns == NUM_COLUMNS_IN_LARGE_NEIGH)

    def test_neigh_metres_to_rowcol_small(self):
        """Ensures correct output from _neigh_metres_to_rowcol.

        In this case the grid is small (5 x 7).
        """

        this_num_rows, this_num_columns = echo_classifn._neigh_metres_to_rowcol(
            neigh_radius_metres=NEIGH_RADIUS_METRES,
            grid_metadata_dict=GRID_METADATA_DICT)

        self.assertTrue(this_num_rows == NUM_ROWS_IN_NEIGH)
        self.assertTrue(this_num_columns == NUM_COLUMNS_IN_NEIGH)

    def test_get_peakedness(self):
        """Ensures correct output from _get_peakedness."""

        this_matrix_dbz = echo_classifn._get_peakedness(
            reflectivity_matrix_dbz=REFLECTIVITY_MATRIX_DBZ,
            num_rows_in_neigh=NUM_ROWS_IN_NEIGH,
            num_columns_in_neigh=NUM_COLUMNS_IN_NEIGH)

        self.assertTrue(numpy.allclose(
            this_matrix_dbz, PEAKEDNESS_MATRIX_DBZ, atol=TOLERANCE))

    def test_apply_convective_criterion1(self):
        """Ensures correct output from _apply_convective_criterion1."""

        this_flag_matrix = echo_classifn._apply_convective_criterion1(
            reflectivity_matrix_dbz=REFLECTIVITY_MATRIX_DBZ,
            peakedness_neigh_metres=NEIGH_RADIUS_METRES,
            max_peakedness_height_m_asl=MAX_PEAKEDNESS_HEIGHT_M_ASL,
            min_composite_refl_dbz=None,
            grid_metadata_dict=GRID_METADATA_DICT)

        self.assertTrue(numpy.array_equal(
            this_flag_matrix, CRITERION1_FLAG_MATRIX))

    def test_apply_convective_criterion2(self):
        """Ensures correct output from _apply_convective_criterion2."""

        this_flag_matrix = echo_classifn._apply_convective_criterion2(
            reflectivity_matrix_dbz=REFLECTIVITY_MATRIX_DBZ,
            convective_flag_matrix=CRITERION1_FLAG_MATRIX,
            grid_metadata_dict=GRID_METADATA_DICT,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC,
            min_composite_refl_aml_dbz=MIN_COMPOSITE_REFL_AML_DBZ)

        self.assertTrue(numpy.array_equal(
            this_flag_matrix, CRITERION2_FLAG_MATRIX))

    def test_apply_convective_criterion3(self):
        """Ensures correct output from _apply_convective_criterion3."""

        this_flag_matrix = echo_classifn._apply_convective_criterion3(
            reflectivity_matrix_dbz=REFLECTIVITY_MATRIX_DBZ,
            convective_flag_matrix=CRITERION2_FLAG_MATRIX,
            grid_metadata_dict=GRID_METADATA_DICT,
            min_echo_top_m_asl=MIN_ECHO_TOP_M_ASL,
            echo_top_level_dbz=ECHO_TOP_LEVEL_DBZ)

        self.assertTrue(numpy.array_equal(
            this_flag_matrix, CRITERION3_FLAG_MATRIX))

    def test_apply_convective_criterion4_main(self):
        """Ensures correct output from _apply_convective_criterion4.

        In this case the input is the "main" flag matrix (the criterion-3 matrix
        created by actually running `_apply_convective_criterion3`).
        """

        this_flag_matrix = echo_classifn._apply_convective_criterion4(
            CRITERION3_FLAG_MATRIX)

        self.assertTrue(numpy.array_equal(
            this_flag_matrix, CRITERION4_FLAG_MATRIX))

    def test_apply_convective_criterion4_dummy(self):
        """Ensures correct output from _apply_convective_criterion4.

        In this case the input is a "dummy" matrix (*not* created by running
        `_apply_convective_criterion3`).
        """

        this_flag_matrix = echo_classifn._apply_convective_criterion4(
            DUMMY_CRITERION3_FLAG_MATRIX)

        self.assertTrue(numpy.array_equal(
            this_flag_matrix, DUMMY_CRITERION4_FLAG_MATRIX))

    def test_apply_convective_criterion5(self):
        """Ensures correct output from _apply_convective_criterion5."""

        this_flag_matrix = echo_classifn._apply_convective_criterion5(
            reflectivity_matrix_dbz=REFLECTIVITY_MATRIX_DBZ,
            convective_flag_matrix=CRITERION4_FLAG_MATRIX,
            min_composite_refl_dbz=MIN_COMPOSITE_REFL_DBZ)

        self.assertTrue(numpy.array_equal(
            this_flag_matrix, CRITERION5_FLAG_MATRIX))

    def test_find_convective_pixels(self):
        """Ensures correct output from find_convective_pixels."""

        this_flag_matrix = echo_classifn.find_convective_pixels(
            reflectivity_matrix_dbz=REFLECTIVITY_MATRIX_DBZ,
            grid_metadata_dict=GRID_METADATA_DICT,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC,
            peakedness_neigh_metres=NEIGH_RADIUS_METRES,
            max_peakedness_height_m_asl=MAX_PEAKEDNESS_HEIGHT_M_ASL,
            min_echo_top_m_asl=MIN_ECHO_TOP_M_ASL,
            echo_top_level_dbz=ECHO_TOP_LEVEL_DBZ,
            min_composite_refl_criterion1_dbz=None,
            min_composite_refl_criterion5_dbz=MIN_COMPOSITE_REFL_DBZ,
            min_composite_refl_aml_dbz=MIN_COMPOSITE_REFL_AML_DBZ)

        self.assertTrue(numpy.array_equal(
            this_flag_matrix, CRITERION5_FLAG_MATRIX))


if __name__ == '__main__':
    unittest.main()
