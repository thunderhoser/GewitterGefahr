"""Unit tests for shape_statistics.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import shape_statistics as shape_stats
from gewittergefahr.gg_utils import polygons

FAKE_STATISTIC_NAME = 'foo'

VERTEX_X_METRES = numpy.array(
    [3., 3., 0., 0., 3., 3., 5., 5., 8., 8., 5., 5., 3.])
VERTEX_Y_METRES = numpy.array(
    [6., 3., 3., 1., 1., 0., 0., 1., 1., 3., 3., 6., 6.])
POLYGON_OBJECT_XY = polygons.vertex_arrays_to_polygon_object(
    VERTEX_X_METRES, VERTEX_Y_METRES)

GRID_SPACING_FOR_BINARY_MATRIX_METRES = 0.5
BINARY_IMAGE_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=bool)

X_OFFSET_METRES = -8.1472
Y_OFFSET_METRES = 9.0579
VERTEX_X_METRES_OFFSET = VERTEX_X_METRES + X_OFFSET_METRES
VERTEX_Y_METRES_OFFSET = VERTEX_Y_METRES + Y_OFFSET_METRES
POLYGON_OBJECT_XY_OFFSET = polygons.vertex_arrays_to_polygon_object(
    VERTEX_X_METRES_OFFSET, VERTEX_Y_METRES_OFFSET)


class ShapeStatisticsTests(unittest.TestCase):
    """Each method is a unit test for shape_statistics.py."""

    def test_check_statistic_names_all_valid(self):
        """Ensures correct output from _check_statistic_names.

        In this case, all input names are valid.
        """

        shape_stats._check_statistic_names(shape_stats.STATISTIC_NAMES)

    def test_check_statistic_names_one_invalid(self):
        """Ensures correct output from _check_statistic_names.

        In this case, one input name is invalid.
        """

        with self.assertRaises(ValueError):
            shape_stats._check_statistic_names(
                shape_stats.STATISTIC_NAMES + [FAKE_STATISTIC_NAME])

    def test_stat_name_new_to_orig_different(self):
        """Ensures correct output from _stat_name_new_to_orig.

        In this case the statistic is area, for which new and original names are
        different.
        """

        this_statistic_name_orig = shape_stats._stat_name_new_to_orig(
            shape_stats.AREA_NAME)
        self.assertTrue(this_statistic_name_orig == shape_stats.AREA_NAME_ORIG)

    def test_stat_name_new_to_orig_same(self):
        """Ensures correct output from _stat_name_new_to_orig.

        In this case the statistic is solidity, for which new and original names
        are the same.
        """

        this_statistic_name_orig = shape_stats._stat_name_new_to_orig(
            shape_stats.SOLIDITY_NAME)
        self.assertTrue(
            this_statistic_name_orig == shape_stats.SOLIDITY_NAME_ORIG)

    def test_get_basic_statistic_names(self):
        """Ensures correct output from _get_basic_statistic_names."""

        these_basic_stat_names = shape_stats._get_basic_statistic_names(
            shape_stats.STATISTIC_NAMES)
        self.assertTrue(set(these_basic_stat_names) ==
                        set(shape_stats.BASIC_STAT_NAMES))

    def test_get_region_property_names(self):
        """Ensures correct output from _get_region_property_names."""

        these_region_prop_names = shape_stats._get_region_property_names(
            shape_stats.STATISTIC_NAMES)
        self.assertTrue(set(these_region_prop_names) ==
                        set(shape_stats.REGION_PROPERTY_NAMES))

    def test_get_curvature_based_stat_names(self):
        """Ensures correct output from _get_curvature_based_stat_names."""

        these_curvature_based_stat_names = (
            shape_stats._get_curvature_based_stat_names(
                shape_stats.STATISTIC_NAMES))

        self.assertTrue(set(these_curvature_based_stat_names) ==
                        set(shape_stats.CURVATURE_BASED_STAT_NAMES))

    def test_xy_polygon_to_binary_matrix_no_offset(self):
        """Ensures correct output from _xy_polygon_to_binary_matrix.

        In this case there is no offset, which means that the minimum x-
        coordinate and minimum y-coordinate over all vertices are zero.
        """

        this_binary_image_matrix = shape_stats._xy_polygon_to_binary_matrix(
            POLYGON_OBJECT_XY,
            grid_spacing_metres=GRID_SPACING_FOR_BINARY_MATRIX_METRES)

        self.assertTrue(numpy.array_equal(
            this_binary_image_matrix, BINARY_IMAGE_MATRIX))


if __name__ == '__main__':
    unittest.main()
