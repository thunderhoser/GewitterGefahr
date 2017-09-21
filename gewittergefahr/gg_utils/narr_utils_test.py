"""Unit tests for narr_utils.py."""

import numpy
import unittest
from gewittergefahr.gg_utils import narr_utils

MAX_MEAN_DISTANCE_ERROR_METRES = 250.
MAX_MAX_DISTANCE_ERROR_METRES = 1000.
GRID_POINT_LATLNG_FILE_NAME = 'grid_point_latlng_grid221.data'


class NarrUtilsTests(unittest.TestCase):
    """Each method is a unit test for narr_utils.py."""

    def test_projection(self):
        """Ensures approximate correctness of Lambert conformal projection.

        NOTE: this is not a unit test.  This method tests the accuracy of the
        Lambert conformal projection generated in `narr_utils.init_projection`.
        This method ensures that the projection (x-y) coordinates of all grid
        points can be generated accurately from lat-long coordinates.
        Specifically, the mean distance error must be <= 250 m and max distance
        error must be <= 1000 m.
        """

        (grid_point_lng_vector_deg, grid_point_lat_vector_deg) = numpy.loadtxt(
            GRID_POINT_LATLNG_FILE_NAME, unpack=True)

        grid_point_lat_matrix_deg = numpy.reshape(grid_point_lat_vector_deg, (
            narr_utils.NUM_GRID_ROWS, narr_utils.NUM_GRID_COLUMNS))
        grid_point_lng_matrix_deg = numpy.reshape(grid_point_lng_vector_deg, (
            narr_utils.NUM_GRID_ROWS, narr_utils.NUM_GRID_COLUMNS))

        (grid_point_x_matrix_metres,
         grid_point_y_matrix_metres) = narr_utils.project_latlng_to_xy(
            grid_point_lat_matrix_deg, grid_point_lng_matrix_deg)

        (expected_grid_point_x_matrix_metres,
         expected_grid_point_y_matrix_metres) = (
            narr_utils.get_xy_grid_point_matrices())

        x_error_matrix_metres = (
            grid_point_x_matrix_metres - expected_grid_point_x_matrix_metres)
        y_error_matrix_metres = (
            grid_point_y_matrix_metres - expected_grid_point_y_matrix_metres)
        distance_error_matrix_metres = numpy.sqrt(
            x_error_matrix_metres ** 2 + y_error_matrix_metres ** 2)

        self.assertTrue(numpy.mean(
            distance_error_matrix_metres) <= MAX_MEAN_DISTANCE_ERROR_METRES)
        self.assertTrue(numpy.max(
            distance_error_matrix_metres) <= MAX_MAX_DISTANCE_ERROR_METRES)


if __name__ == '__main__':
    unittest.main()
