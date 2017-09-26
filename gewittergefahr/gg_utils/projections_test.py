"""Unit tests for projections.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import projections

TOLERANCE = 1e-6

STANDARD_LATITUDES_DEG = numpy.array([50., 50.])
CENTRAL_LATITUDE_DEG = 40.
CENTRAL_LONGITUDE_DEG = 253.
FALSE_EASTING_METRES = 0.
FALSE_NORTHING_METRES = 0.

LATITUDES_DEG = numpy.array([51.05, 53.55, 52.27, 50.05, 49.7])
LONGITUDES_DEG = numpy.array([245.95, 246.53, 246.2, 249.33, 247.18])
X_COORDS_METRES = numpy.array([100., 500., 1000., 10000., -3000.])
Y_COORDS_METRES = numpy.array([-6000., 7500., 8000., 900., -5.])


class ProjectionsTests(unittest.TestCase):
    """Each method is a unit test for projections.py."""

    def test_init_lambert_conformal_projection_no_crash(self):
        """Ensures that init_lambert_conformal_projection does not crash."""

        projections.init_lambert_conformal_projection(STANDARD_LATITUDES_DEG,
                                                      CENTRAL_LONGITUDE_DEG)

    def test_init_azimuthal_equidistant_projection_no_crash(self):
        """Ensures that init_azimuthal_equidistant_projection does not crash."""

        projections.init_azimuthal_equidistant_projection(CENTRAL_LATITUDE_DEG,
                                                          CENTRAL_LONGITUDE_DEG)

    def test_project_latlng_to_xy(self):
        """Ensures that project_latlng_to_xy does not crash.

        This is an integration test, not a unit test, because it requires
        init_lambert_conformal_projection to create the projection object.
        """

        projection_object = projections.init_lambert_conformal_projection(
            STANDARD_LATITUDES_DEG, CENTRAL_LONGITUDE_DEG)

        projections.project_latlng_to_xy(
            LATITUDES_DEG, LONGITUDES_DEG, projection_object=projection_object,
            false_easting_metres=FALSE_EASTING_METRES,
            false_northing_metres=FALSE_NORTHING_METRES)

    def test_project_xy_to_latlng(self):
        """Ensures that project_xy_to_latlng does not crash.

        This is an integration test, not a unit test, because it requires
        init_lambert_conformal_projection to create the projection object.
        """

        projection_object = projections.init_lambert_conformal_projection(
            STANDARD_LATITUDES_DEG, CENTRAL_LONGITUDE_DEG)

        projections.project_xy_to_latlng(
            X_COORDS_METRES, Y_COORDS_METRES,
            projection_object=projection_object,
            false_easting_metres=FALSE_EASTING_METRES,
            false_northing_metres=FALSE_NORTHING_METRES)

    def test_project_both_ways(self):
        """Ensures that the two projection methods are inverses.

        This is an integration test, not a unit test, because it calls both
        projection methods.  Also, it requires init_lambert_conformal_projection
        to create the projection object.
        """

        projection_object = projections.init_lambert_conformal_projection(
            STANDARD_LATITUDES_DEG, CENTRAL_LONGITUDE_DEG)

        (these_x_coords_metres,
         these_y_coords_metres) = projections.project_latlng_to_xy(
             LATITUDES_DEG, LONGITUDES_DEG, projection_object=projection_object,
             false_easting_metres=FALSE_EASTING_METRES,
             false_northing_metres=FALSE_NORTHING_METRES)

        (these_latitudes_deg,
         these_longitudes_deg) = projections.project_xy_to_latlng(
             these_x_coords_metres, these_y_coords_metres,
             projection_object=projection_object,
             false_easting_metres=FALSE_EASTING_METRES,
             false_northing_metres=FALSE_NORTHING_METRES)

        self.assertTrue(
            numpy.allclose(these_latitudes_deg, LATITUDES_DEG, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(these_longitudes_deg, LONGITUDES_DEG,
                                       atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
