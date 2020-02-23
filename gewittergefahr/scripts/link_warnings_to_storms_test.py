"""Unit tests for link_warnings_to_storms.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import linkage_test
from gewittergefahr.scripts import link_warnings_to_storms as link_warnings

TOLERANCE = 1e-6
STORM_OBJECT_TABLE = linkage_test.create_storm_objects()

# The following constants are used to test _find_one_polygon_distance.
RELATIVE_X_VERTICES_METRES = numpy.array([-0.5, 0, 0.5, 0, -0.5])
RELATIVE_Y_VERTICES_METRES = numpy.array([0, 0.5, 0, -0.5, 0])

STORM_TO_X_VERTICES_METRES = {
    0: RELATIVE_X_VERTICES_METRES - 10,
    1: RELATIVE_X_VERTICES_METRES - 15,
    2: RELATIVE_X_VERTICES_METRES,
    3: RELATIVE_X_VERTICES_METRES,
    4: RELATIVE_X_VERTICES_METRES + 10,
    5: RELATIVE_X_VERTICES_METRES + 15,
    6: RELATIVE_X_VERTICES_METRES,
    7: RELATIVE_X_VERTICES_METRES,
    8: RELATIVE_X_VERTICES_METRES - 15,
}

STORM_TO_Y_VERTICES_METRES = {
    0: RELATIVE_Y_VERTICES_METRES,
    1: RELATIVE_Y_VERTICES_METRES,
    2: RELATIVE_Y_VERTICES_METRES + 3,
    3: RELATIVE_Y_VERTICES_METRES + 5,
    4: RELATIVE_Y_VERTICES_METRES,
    5: RELATIVE_Y_VERTICES_METRES,
    6: RELATIVE_Y_VERTICES_METRES - 3,
    7: RELATIVE_Y_VERTICES_METRES - 5,
    8: RELATIVE_Y_VERTICES_METRES - 5,
}

STORM_TO_WARNING_DIST_METRES = {
    0: 0.,
    1: 4.5,
    2: 0.,
    3: 1.5,
    4: 0.,
    5: 4.5,
    6: 0.,
    7: 1.5,
    8: numpy.sqrt(24.25),
}

THESE_X_METRES = numpy.array([-10, -10, 10, 10, -10], dtype=float)
THESE_Y_METRES = numpy.array([-3, 3, 3, -3, -3], dtype=float)
WARNING_POLYGON_OBJECT_XY = polygons.vertex_arrays_to_polygon_object(
    exterior_x_coords=THESE_X_METRES, exterior_y_coords=THESE_Y_METRES
)

# The following constants are used to test _link_one_warning.
THESE_START_TIMES_UNIX_SEC = numpy.array([4, 3], dtype=int)
THESE_END_TIMES_UNIX_SEC = numpy.array([6, 5], dtype=int)

THESE_X_METRES = numpy.array([274, 274, 276, 276, 274], dtype=float)
THESE_Y_METRES = numpy.array([59.5, 60.5, 60.5, 59.5, 59.5])
FIRST_POLYGON_OBJECT = polygons.vertex_arrays_to_polygon_object(
    exterior_x_coords=THESE_X_METRES, exterior_y_coords=THESE_Y_METRES
)

THESE_X_METRES = numpy.array([273, 273, 275, 275, 273], dtype=float)
SECOND_POLYGON_OBJECT = polygons.vertex_arrays_to_polygon_object(
    exterior_x_coords=THESE_X_METRES, exterior_y_coords=THESE_Y_METRES
)

THIS_DICT = {
    link_warnings.WARNING_START_TIME_KEY: THESE_START_TIMES_UNIX_SEC,
    link_warnings.WARNING_END_TIME_KEY: THESE_END_TIMES_UNIX_SEC,
    link_warnings.WARNING_LATLNG_POLYGON_KEY:
        [FIRST_POLYGON_OBJECT, SECOND_POLYGON_OBJECT],
    link_warnings.WARNING_XY_POLYGON_KEY:
        [FIRST_POLYGON_OBJECT, SECOND_POLYGON_OBJECT]
}

WARNING_TABLE = pandas.DataFrame.from_dict(THIS_DICT)

MAX_DISTANCE_METRES = 1.
MIN_LIFETIME_FRACTION = 0.5
WARNING_TO_SEC_ID_STRINGS = {
    0: ['B4', 'B5'],
    1: ['B4']
}


class LinkWarningsToStormsTests(unittest.TestCase):
    """Each method is a unit test for link_warnings_to_storms.py."""

    def test_find_one_polygon_distance(self):
        """Ensures correct output from _find_one_polygon_distance."""

        num_storms = len(STORM_TO_X_VERTICES_METRES)

        for i in range(num_storms):
            this_distance_metres = link_warnings._find_one_polygon_distance(
                storm_x_vertices_metres=STORM_TO_X_VERTICES_METRES[i],
                storm_y_vertices_metres=STORM_TO_Y_VERTICES_METRES[i],
                warning_polygon_object_xy=WARNING_POLYGON_OBJECT_XY
            )

            self.assertTrue(numpy.isclose(
                this_distance_metres, STORM_TO_WARNING_DIST_METRES[i],
                atol=TOLERANCE
            ))

    def test_link_one_warning_first(self):
        """Ensures correct output from _link_one_warning.

        In this case, trying to link first warning.
        """

        actual_sec_id_strings = link_warnings._link_one_warning(
            warning_table=WARNING_TABLE.iloc[[0]],
            storm_object_table=copy.deepcopy(STORM_OBJECT_TABLE),
            max_distance_metres=MAX_DISTANCE_METRES,
            min_lifetime_fraction=MIN_LIFETIME_FRACTION, test_mode=True
        )

        expected_sec_id_strings = WARNING_TO_SEC_ID_STRINGS[0]
        self.assertTrue(
            set(expected_sec_id_strings) == set(actual_sec_id_strings)
        )

    def test_link_one_warning_second(self):
        """Ensures correct output from _link_one_warning.

        In this case, trying to link second warning.
        """

        actual_sec_id_strings = link_warnings._link_one_warning(
            warning_table=WARNING_TABLE.iloc[[1]],
            storm_object_table=copy.deepcopy(STORM_OBJECT_TABLE),
            max_distance_metres=MAX_DISTANCE_METRES,
            min_lifetime_fraction=MIN_LIFETIME_FRACTION, test_mode=True
        )

        expected_sec_id_strings = WARNING_TO_SEC_ID_STRINGS[1]
        self.assertTrue(
            set(expected_sec_id_strings) == set(actual_sec_id_strings)
        )


if __name__ == '__main__':
    unittest.main()
