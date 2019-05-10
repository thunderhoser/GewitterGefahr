"""Unit tests for projections.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import projections

TOLERANCE = 1e-6

STANDARD_LATITUDES_DEG = numpy.full(2, 50.)
CENTRAL_LATITUDE_DEG = 40.
CENTRAL_LONGITUDE_DEG = 253.
FALSE_EASTING_METRES = 0.
FALSE_NORTHING_METRES = 0.

LATITUDES_DEG = numpy.array([
    [42, -35, numpy.nan],
    [61, -33, -30],
    [-44, 39, 44],
    [33, -46, numpy.nan]
])

LONGITUDES_DEG = numpy.array([
    [287, 254, numpy.nan],
    [211, 276, 97],
    [117, 284, 68],
    [86, 260, numpy.nan]
])

X_COORDS_METRES = numpy.array([
    [4045, 539, numpy.nan],
    [1354, 4523, 1644],
    [4750, 1838, 4847],
    [3050, 4847, numpy.nan]
])

Y_COORDS_METRES = numpy.array([
    [4540, 3905, numpy.nan],
    [5413, 2345, 4416],
    [5047, 3818, 4748],
    [5030, 4748, numpy.nan]
])


class ProjectionsTests(unittest.TestCase):
    """Each method is a unit test for projections.py."""

    def test_init_lcc_projection_no_crash(self):
        """Ensures that init_lcc_projection does not crash."""

        projections.init_lcc_projection(
            standard_latitudes_deg=STANDARD_LATITUDES_DEG,
            central_longitude_deg=CENTRAL_LONGITUDE_DEG)

    def test_init_azimuthal_equidistant_projection_no_crash(self):
        """Ensures that init_azimuthal_equidistant_projection does not crash."""

        projections.init_azimuthal_equidistant_projection(
            central_latitude_deg=CENTRAL_LATITUDE_DEG,
            central_longitude_deg=CENTRAL_LONGITUDE_DEG)

    def test_project_latlng_to_xy(self):
        """Ensures that project_latlng_to_xy does not crash.

        This is an integration test, not a unit test, because it requires
        init_lcc_projection to create the projection object.
        """

        projection_object = projections.init_lcc_projection(
            standard_latitudes_deg=STANDARD_LATITUDES_DEG,
            central_longitude_deg=CENTRAL_LONGITUDE_DEG)

        projections.project_latlng_to_xy(
            latitudes_deg=LATITUDES_DEG, longitudes_deg=LONGITUDES_DEG,
            projection_object=projection_object,
            false_easting_metres=FALSE_EASTING_METRES,
            false_northing_metres=FALSE_NORTHING_METRES)

    def test_project_xy_to_latlng(self):
        """Ensures that project_xy_to_latlng does not crash.

        This is an integration test, not a unit test, because it requires
        init_lcc_projection to create the projection object.
        """

        projection_object = projections.init_lcc_projection(
            standard_latitudes_deg=STANDARD_LATITUDES_DEG,
            central_longitude_deg=CENTRAL_LONGITUDE_DEG)

        projections.project_xy_to_latlng(
            x_coords_metres=X_COORDS_METRES, y_coords_metres=Y_COORDS_METRES,
            projection_object=projection_object,
            false_easting_metres=FALSE_EASTING_METRES,
            false_northing_metres=FALSE_NORTHING_METRES)

    def test_project_both_ways(self):
        """Ensures that the two projection methods are inverses.

        This is an integration test, not a unit test, because it calls both
        projection methods.  Also, it requires init_lcc_projection
        to create the projection object.
        """

        projection_object = projections.init_lcc_projection(
            standard_latitudes_deg=STANDARD_LATITUDES_DEG,
            central_longitude_deg=CENTRAL_LONGITUDE_DEG)

        these_x_coords_metres, these_y_coords_metres = (
            projections.project_latlng_to_xy(
                latitudes_deg=LATITUDES_DEG, longitudes_deg=LONGITUDES_DEG,
                projection_object=projection_object,
                false_easting_metres=FALSE_EASTING_METRES,
                false_northing_metres=FALSE_NORTHING_METRES)
        )

        these_latitudes_deg, these_longitudes_deg = (
            projections.project_xy_to_latlng(
                x_coords_metres=these_x_coords_metres,
                y_coords_metres=these_y_coords_metres,
                projection_object=projection_object,
                false_easting_metres=FALSE_EASTING_METRES,
                false_northing_metres=FALSE_NORTHING_METRES)
        )

        self.assertTrue(numpy.allclose(
            these_latitudes_deg, LATITUDES_DEG, atol=TOLERANCE, equal_nan=True
        ))
        self.assertTrue(numpy.allclose(
            these_longitudes_deg, LONGITUDES_DEG, atol=TOLERANCE,
            equal_nan=True
        ))


if __name__ == '__main__':
    unittest.main()
