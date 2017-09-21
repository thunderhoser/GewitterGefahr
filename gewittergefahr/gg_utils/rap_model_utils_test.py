"""Unit tests for rap_model_utils.py."""

import numpy
import unittest
from gewittergefahr.gg_utils import rap_model_utils

MAX_MEAN_DISTANCE_ERROR_METRES = 100.
MAX_MAX_DISTANCE_ERROR_METRES = 500.
GRID130_LATLNG_FILE_NAME = 'grid_point_latlng_grid130.data'
GRID252_LATLNG_FILE_NAME = 'grid_point_latlng_grid252.data'


class RapModelUtilsTests(unittest.TestCase):
    """Each method is a unit test for rap_model_utils.py."""

    def test_projection_grid130(self):
        """Ensures approx correctness of Lambert projection for grid 130.

        NOTE: this is not a unit test.  This method tests the accuracy of the
        Lambert conformal projection generated in
        `rap_model_utils.init_projection` for grid 130.  This method ensures
        that the projection (x-y) coordinates of all grid points can be
        generated accurately from lat-long coordinates.  Specifically, the mean
        distance error must be <= 100 m and max distance error must be <= 500 m.
        """

        (grid_point_lng_vector_deg, grid_point_lat_vector_deg) = numpy.loadtxt(
            GRID130_LATLNG_FILE_NAME, unpack=True)

        grid_point_lat_matrix_deg = numpy.reshape(grid_point_lat_vector_deg, (
            rap_model_utils.NUM_ROWS_130GRID,
            rap_model_utils.NUM_COLUMNS_130GRID))
        grid_point_lng_matrix_deg = numpy.reshape(grid_point_lng_vector_deg, (
            rap_model_utils.NUM_ROWS_130GRID,
            rap_model_utils.NUM_COLUMNS_130GRID))

        (grid_point_x_matrix_metres,
         grid_point_y_matrix_metres) = rap_model_utils.project_latlng_to_xy(
            grid_point_lat_matrix_deg, grid_point_lng_matrix_deg,
            grid_id=rap_model_utils.ID_FOR_130GRID)

        (expected_grid_point_x_matrix_metres,
         expected_grid_point_y_matrix_metres) = (
            rap_model_utils.get_xy_grid_point_matrices(
                grid_id=rap_model_utils.ID_FOR_130GRID))

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

    def test_projection_grid252(self):
        """Same as test_projection_grid130, but for grid 252 instead of 130."""

        (grid_point_lng_vector_deg, grid_point_lat_vector_deg) = numpy.loadtxt(
            GRID252_LATLNG_FILE_NAME, unpack=True)

        grid_point_lat_matrix_deg = numpy.reshape(grid_point_lat_vector_deg, (
            rap_model_utils.NUM_ROWS_252GRID,
            rap_model_utils.NUM_COLUMNS_252GRID))
        grid_point_lng_matrix_deg = numpy.reshape(grid_point_lng_vector_deg, (
            rap_model_utils.NUM_ROWS_252GRID,
            rap_model_utils.NUM_COLUMNS_252GRID))

        (grid_point_x_matrix_metres,
         grid_point_y_matrix_metres) = rap_model_utils.project_latlng_to_xy(
            grid_point_lat_matrix_deg, grid_point_lng_matrix_deg,
            grid_id=rap_model_utils.ID_FOR_252GRID)

        (expected_grid_point_x_matrix_metres,
         expected_grid_point_y_matrix_metres) = (
            rap_model_utils.get_xy_grid_point_matrices(
                grid_id=rap_model_utils.ID_FOR_252GRID))

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
