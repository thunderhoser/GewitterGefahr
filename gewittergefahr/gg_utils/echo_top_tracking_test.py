"""Unit tests for echo_top_tracking.py."""

import copy
import unittest
import numpy
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import temporal_tracking
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import projections

TOLERANCE = 1e-6
RELATIVE_DISTANCE_TOLERANCE = 0.015

# The following constants are used to test _find_local_maxima.
RADAR_MATRIX = numpy.array([
    [0, numpy.nan, 3, 4, numpy.nan, 6],
    [7, 8, 9, 10, numpy.nan, numpy.nan],
    [13, 14, numpy.nan, numpy.nan, 17, 18],
    [19, 20, numpy.nan, numpy.nan, numpy.nan, 24],
    [numpy.nan, numpy.nan, 27, 28, 29, 30]
])

RADAR_METADATA_DICT = {
    radar_utils.NW_GRID_POINT_LAT_COLUMN: 35.,
    radar_utils.NW_GRID_POINT_LNG_COLUMN: 95.,
    radar_utils.LAT_SPACING_COLUMN: 0.01,
    radar_utils.LNG_SPACING_COLUMN: 0.02
}

NEIGH_HALF_WIDTH_PIXELS = 1

LOCAL_MAX_ROWS = numpy.array([0, 4], dtype=int)
LOCAL_MAX_COLUMNS = numpy.array([5, 5], dtype=int)
LOCAL_MAX_LATITUDES_DEG = numpy.array([34.96, 35])
LOCAL_MAX_LONGITUDES_DEG = numpy.array([95.1, 95.1])
LOCAL_MAX_VALUES = numpy.array([30, 6], dtype=float)

LOCAL_MAX_DICT_LATLNG = {
    temporal_tracking.LATITUDES_KEY: LOCAL_MAX_LATITUDES_DEG,
    temporal_tracking.LONGITUDES_KEY: LOCAL_MAX_LONGITUDES_DEG,
    echo_top_tracking.MAX_VALUES_KEY: LOCAL_MAX_VALUES
}

# The following constants are used to test _remove_redundant_local_maxima.
SMALL_INTERMAX_DISTANCE_METRES = 1000.
LARGE_INTERMAX_DISTANCE_METRES = 10000.

PROJECTION_OBJECT = projections.init_azimuthal_equidistant_projection(
    central_latitude_deg=35., central_longitude_deg=95.)

LOCAL_MAX_X_COORDS_METRES, LOCAL_MAX_Y_COORDS_METRES = (
    projections.project_latlng_to_xy(
        LOCAL_MAX_LATITUDES_DEG, LOCAL_MAX_LONGITUDES_DEG,
        projection_object=PROJECTION_OBJECT,
        false_easting_metres=0., false_northing_metres=0.)
)

LOCAL_MAX_DICT_SMALL_DISTANCE = {
    temporal_tracking.LATITUDES_KEY: LOCAL_MAX_LATITUDES_DEG,
    temporal_tracking.LONGITUDES_KEY: LOCAL_MAX_LONGITUDES_DEG,
    echo_top_tracking.MAX_VALUES_KEY: LOCAL_MAX_VALUES,
    temporal_tracking.X_COORDS_KEY: LOCAL_MAX_X_COORDS_METRES,
    temporal_tracking.Y_COORDS_KEY: LOCAL_MAX_Y_COORDS_METRES
}

LOCAL_MAX_DICT_LARGE_DISTANCE = {
    temporal_tracking.LATITUDES_KEY: LOCAL_MAX_LATITUDES_DEG[:-1],
    temporal_tracking.LONGITUDES_KEY: LOCAL_MAX_LONGITUDES_DEG[:-1],
    echo_top_tracking.MAX_VALUES_KEY: LOCAL_MAX_VALUES[:-1],
    temporal_tracking.X_COORDS_KEY: LOCAL_MAX_X_COORDS_METRES[:-1],
    temporal_tracking.Y_COORDS_KEY: LOCAL_MAX_Y_COORDS_METRES[:-1]
}

# The following constants are used to test _remove_small_polygons.
THIS_LIST_OF_ROW_ARRAYS = [
    numpy.array([0, 0, 0, 0, 1, 1, 2, 2, 2], dtype=int),
    numpy.array([-5, -4, -3], dtype=int),
    numpy.array([0, 1, 1, 2, 3, 5, 8, 13, 6, 6, 6], dtype=int)
]

LOCAL_MAX_DICT_WITH_SMALL = {
    temporal_tracking.GRID_POINT_ROWS_KEY: THIS_LIST_OF_ROW_ARRAYS,
    temporal_tracking.LATITUDES_KEY: numpy.array([51.1, 53.5, 60]),
    temporal_tracking.LONGITUDES_KEY: numpy.array([246, 246.5, 250])
}

MIN_POLYGON_SIZE_PIXELS = 5
THIS_LIST_OF_ROW_ARRAYS = [
    numpy.array([0, 0, 0, 0, 1, 1, 2, 2, 2], dtype=int),
    numpy.array([0, 1, 1, 2, 3, 5, 8, 13, 6, 6, 6], dtype=int)
]

LOCAL_MAX_DICT_WITHOUT_SMALL = {
    temporal_tracking.GRID_POINT_ROWS_KEY: THIS_LIST_OF_ROW_ARRAYS,
    temporal_tracking.LATITUDES_KEY: numpy.array([51.1, 60]),
    temporal_tracking.LONGITUDES_KEY: numpy.array([246, 250])
}


def _compare_local_max_dicts(first_local_max_dict, second_local_max_dict):
    """Compares two dictionaries with local maxima.

    :param first_local_max_dict: First dictionary.
    :param second_local_max_dict: Second dictionary.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = first_local_max_dict.keys()
    second_keys = second_local_max_dict.keys()
    if set(first_keys) != set(second_keys):
        return False

    for this_key in first_keys:
        if this_key == temporal_tracking.GRID_POINT_ROWS_KEY:
            first_length = len(first_local_max_dict[this_key])
            second_length = len(second_local_max_dict[this_key])
            if first_length != second_length:
                return False

            for i in range(first_length):
                if not numpy.array_equal(first_local_max_dict[this_key][i],
                                         second_local_max_dict[this_key][i]):
                    return False

        else:
            if not numpy.allclose(first_local_max_dict[this_key],
                                  second_local_max_dict[this_key],
                                  atol=TOLERANCE):
                return False

    return True


class EchoTopTrackingTests(unittest.TestCase):
    """Each method is a unit test for echo_top_tracking.py."""

    def test_find_local_maxima(self):
        """Ensures correct output from _find_local_maxima."""

        this_local_max_dict = echo_top_tracking._find_local_maxima(
            radar_matrix=RADAR_MATRIX, radar_metadata_dict=RADAR_METADATA_DICT,
            neigh_half_width_pixels=NEIGH_HALF_WIDTH_PIXELS)

        these_keys = set(list(this_local_max_dict))
        expected_keys = set(list(LOCAL_MAX_DICT_LATLNG))
        self.assertTrue(these_keys == expected_keys)

        for this_key in these_keys:
            self.assertTrue(numpy.allclose(
                this_local_max_dict[this_key], LOCAL_MAX_DICT_LATLNG[this_key],
                atol=TOLERANCE
            ))

    def test_remove_redundant_local_maxima_small_distance(self):
        """Ensures correct output from _remove_redundant_local_maxima.

        In this case, minimum distance between two maxima is small.
        """

        this_local_max_dict = echo_top_tracking._remove_redundant_local_maxima(
            local_max_dict=copy.deepcopy(LOCAL_MAX_DICT_LATLNG),
            projection_object=PROJECTION_OBJECT,
            min_intermax_distance_metres=SMALL_INTERMAX_DISTANCE_METRES)

        these_keys = set(list(this_local_max_dict))
        expected_keys = set(list(LOCAL_MAX_DICT_SMALL_DISTANCE))
        self.assertTrue(these_keys == expected_keys)

        for this_key in these_keys:
            self.assertTrue(numpy.allclose(
                this_local_max_dict[this_key],
                LOCAL_MAX_DICT_SMALL_DISTANCE[this_key], atol=TOLERANCE
            ))

    def test_remove_redundant_local_maxima_large_distance(self):
        """Ensures correct output from _remove_redundant_local_maxima.

        In this case, minimum distance between two maxima is large.
        """

        this_local_max_dict = echo_top_tracking._remove_redundant_local_maxima(
            local_max_dict=copy.deepcopy(LOCAL_MAX_DICT_LATLNG),
            projection_object=PROJECTION_OBJECT,
            min_intermax_distance_metres=LARGE_INTERMAX_DISTANCE_METRES)

        these_keys = set(list(this_local_max_dict))
        expected_keys = set(list(LOCAL_MAX_DICT_LARGE_DISTANCE))
        self.assertTrue(these_keys == expected_keys)

        for this_key in these_keys:
            self.assertTrue(numpy.allclose(
                this_local_max_dict[this_key],
                LOCAL_MAX_DICT_LARGE_DISTANCE[this_key], atol=TOLERANCE
            ))

    def test_remove_small_polygons_min0(self):
        """Ensures correct output from _remove_small_polygons.

        In this case polygons with >= 0 grid cells should be kept.
        """

        this_local_max_dict = echo_top_tracking._remove_small_polygons(
            local_max_dict=copy.deepcopy(LOCAL_MAX_DICT_WITH_SMALL),
            min_size_pixels=0)

        self.assertTrue(_compare_local_max_dicts(
            this_local_max_dict, LOCAL_MAX_DICT_WITH_SMALL
        ))

    def test_remove_small_polygons_min5(self):
        """Ensures correct output from _remove_small_polygons.

        In this case polygons with >= 5 grid cells should be kept.
        """

        this_local_max_dict = echo_top_tracking._remove_small_polygons(
            local_max_dict=copy.deepcopy(LOCAL_MAX_DICT_WITH_SMALL),
            min_size_pixels=MIN_POLYGON_SIZE_PIXELS)

        self.assertTrue(_compare_local_max_dicts(
            this_local_max_dict, LOCAL_MAX_DICT_WITHOUT_SMALL
        ))


if __name__ == '__main__':
    unittest.main()
