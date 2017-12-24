"""Unit tests for gridded_forecasts.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import gridded_forecasts

TOLERANCE = 1e-6

# The following constants are used to test _column_name_to_distance_buffer,
# _distance_buffer_to_column_name, _get_distance_buffer_columns, and
# _check_distance_buffers.
SMALL_BUFFER_MIN_DISTANCE_METRES = numpy.nan
SMALL_BUFFER_MAX_DISTANCE_METRES = 0.
SMALL_BUFFER_LATLNG_COLUMN = 'polygon_object_latlng_buffer_0m'
SMALL_BUFFER_XY_COLUMN = 'polygon_object_xy_buffer_0m'
SMALL_BUFFER_FORECAST_COLUMN = 'forecast_probability_buffer_0m'
SMALL_BUFFER_GRID_ROWS_COLUMN = 'grid_rows_in_buffer_0m'
SMALL_BUFFER_GRID_COLUMNS_COLUMN = 'grid_columns_in_buffer_0m'

MEDIUM_BUFFER_MIN_DISTANCE_METRES = 0.
MEDIUM_BUFFER_MAX_DISTANCE_METRES = 5000.

LARGE_BUFFER_MIN_DISTANCE_METRES = 5000.
LARGE_BUFFER_MAX_DISTANCE_METRES = 10000.
LARGE_BUFFER_LATLNG_COLUMN = 'polygon_object_latlng_buffer_5000m_10000m'
LARGE_BUFFER_XY_COLUMN = 'polygon_object_xy_buffer_5000m_10000m'
LARGE_BUFFER_FORECAST_COLUMN = 'forecast_probability_buffer_5000m_10000m'
LARGE_BUFFER_GRID_ROWS_COLUMN = 'grid_rows_in_buffer_5000m_10000m'
LARGE_BUFFER_GRID_COLUMNS_COLUMN = 'grid_columns_in_buffer_5000m_10000m'

EMPTY_STORM_OBJECT_DICT = {
    SMALL_BUFFER_LATLNG_COLUMN: [], LARGE_BUFFER_LATLNG_COLUMN: [],
    SMALL_BUFFER_XY_COLUMN: [], LARGE_BUFFER_XY_COLUMN: [],
    SMALL_BUFFER_FORECAST_COLUMN: [], LARGE_BUFFER_FORECAST_COLUMN: [],
    SMALL_BUFFER_GRID_ROWS_COLUMN: [], LARGE_BUFFER_GRID_ROWS_COLUMN: [],
    SMALL_BUFFER_GRID_COLUMNS_COLUMN: [], LARGE_BUFFER_GRID_COLUMNS_COLUMN: []
}
EMPTY_STORM_OBJECT_TABLE = pandas.DataFrame.from_dict(EMPTY_STORM_OBJECT_DICT)

# The following constants are used to test _create_xy_grid and
# _normalize_probs_by_polygon_area.
SMALL_BUFFER_VERTEX_X_METRES = numpy.array([-10., -5., -5., -10., -10.])
SMALL_BUFFER_VERTEX_Y_METRES = numpy.array([-20., -20., -10., -10., -20.])
SMALL_BUFFER_AREA_METRES2 = 50.
SMALL_BUFFER_FORECAST_PROBS = numpy.array(
    [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.])

LARGE_BUFFER_VERTEX_X_METRES = numpy.array([0., 10., 10., 0., 0.])
LARGE_BUFFER_VERTEX_Y_METRES = numpy.array([0., 0., 20., 20., 0.])
LARGE_BUFFER_AREA_METRES2 = 200.
LARGE_BUFFER_FORECAST_PROBS = numpy.array(
    [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.])

PROB_AREA_FOR_GRID_METRES2 = 100.
PROB_RADIUS_FOR_GRID_METRES = numpy.sqrt(PROB_AREA_FOR_GRID_METRES2 / numpy.pi)
SMALL_BUFFER_NORMALIZED_PROBS = numpy.array(
    [0., 0.2, 0.4, 0.6, 0.8, 1., 1., 1.])
LARGE_BUFFER_NORMALIZED_PROBS = numpy.array(
    [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.375, 0.5])

SMALL_BUFFER_POLYGON_OBJECT_XY = polygons.vertex_arrays_to_polygon_object(
    SMALL_BUFFER_VERTEX_X_METRES, SMALL_BUFFER_VERTEX_Y_METRES)
LARGE_BUFFER_POLYGON_OBJECT_XY = polygons.vertex_arrays_to_polygon_object(
    LARGE_BUFFER_VERTEX_X_METRES, LARGE_BUFFER_VERTEX_Y_METRES)

SMALL_OBJECT_ARRAY = numpy.full(
    len(SMALL_BUFFER_FORECAST_PROBS), SMALL_BUFFER_POLYGON_OBJECT_XY,
    dtype=object)
LARGE_OBJECT_ARRAY = numpy.full(
    len(SMALL_BUFFER_FORECAST_PROBS), LARGE_BUFFER_POLYGON_OBJECT_XY,
    dtype=object)

THIS_DICT = {SMALL_BUFFER_XY_COLUMN: SMALL_OBJECT_ARRAY,
             LARGE_BUFFER_XY_COLUMN: LARGE_OBJECT_ARRAY,
             SMALL_BUFFER_FORECAST_COLUMN: SMALL_BUFFER_FORECAST_PROBS,
             LARGE_BUFFER_FORECAST_COLUMN: LARGE_BUFFER_FORECAST_PROBS}
STORM_OBJECT_TABLE = pandas.DataFrame.from_dict(THIS_DICT)

GRID_SPACING_X_METRES = 10.
GRID_SPACING_Y_METRES = 15.
MAX_LEAD_TIME_SEC = 1
GRID_POINTS_X_METRES = numpy.array(
    [-70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60.,
     70.])
GRID_POINTS_Y_METRES = numpy.array(
    [-90., -75., -60., -45., -30., -15., 0., 15., 30., 45., 60., 75., 90.])

# The following constants are used to test _create_latlng_grid.
PROJECTION_OBJECT = projections.init_azimuthal_equidistant_projection(
    central_latitude_deg=35., central_longitude_deg=265.)
GRID_LAT_SPACING_DEG = 0.0002
GRID_LNG_SPACING_DEG = 0.0001

GRID_POINT_LATITUDES_DEG = numpy.array(
    [34.999, 34.9992, 34.9994, 34.9996, 34.9998, 35., 35.0002, 35.0004, 35.0006,
     35.0008, 35.001])
GRID_POINT_LONGITUDES_DEG = numpy.array(
    [264.9992, 264.9993, 264.9994, 264.9995, 264.9996, 264.9997, 264.9998,
     264.9999, 265., 265.0001, 265.0002, 265.0003, 265.0004, 265.0005, 265.0006,
     265.0007, 265.0008])

# The following constants are used to test _find_grid_points_in_polygon.
GRID_SPACING_FOR_PIP_X_METRES = 2.
GRID_SPACING_FOR_PIP_Y_METRES = 3.
MIN_GRID_POINT_FOR_PIP_X_METRES = -20.
MIN_GRID_POINT_FOR_PIP_Y_METRES = -40.
NUM_GRID_ROWS_FOR_PIP = 25
NUM_GRID_COLUMNS_FOR_PIP = 20

GRID_POINTS_FOR_PIP_X_METRES = numpy.linspace(
    MIN_GRID_POINT_FOR_PIP_X_METRES,
    MIN_GRID_POINT_FOR_PIP_X_METRES + (
        (NUM_GRID_COLUMNS_FOR_PIP - 1) * GRID_SPACING_FOR_PIP_X_METRES),
    num=NUM_GRID_COLUMNS_FOR_PIP)
GRID_POINTS_FOR_PIP_Y_METRES = numpy.linspace(
    MIN_GRID_POINT_FOR_PIP_Y_METRES,
    MIN_GRID_POINT_FOR_PIP_Y_METRES + (
        (NUM_GRID_ROWS_FOR_PIP - 1) * GRID_SPACING_FOR_PIP_Y_METRES),
    num=NUM_GRID_ROWS_FOR_PIP)

GRID_ROWS_IN_SMALL_BUFFER = numpy.array(
    [7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10], dtype=int)
GRID_COLUMNS_IN_SMALL_BUFFER = numpy.array(
    [5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7], dtype=int)

GRID_ROWS_IN_LARGE_BUFFER = numpy.array(
    [14, 14, 14, 14, 14, 14,
     15, 15, 15, 15, 15, 15,
     16, 16, 16, 16, 16, 16,
     17, 17, 17, 17, 17, 17,
     18, 18, 18, 18, 18, 18,
     19, 19, 19, 19, 19, 19,
     20, 20, 20, 20, 20, 20], dtype=int)
GRID_COLUMNS_IN_LARGE_BUFFER = numpy.array(
    [10, 11, 12, 13, 14, 15,
     10, 11, 12, 13, 14, 15,
     10, 11, 12, 13, 14, 15,
     10, 11, 12, 13, 14, 15,
     10, 11, 12, 13, 14, 15,
     10, 11, 12, 13, 14, 15,
     10, 11, 12, 13, 14, 15], dtype=int)

# The following constants are used to test _find_min_value_greater_or_equal and
# _find_max_value_less_than_or_equal.
SORTED_ARRAY = numpy.array([-4., -2., 0., 2., 5., 8.])
SMALL_TEST_VALUE_IN_ARRAY = -2.
SMALL_TEST_VALUE_NOT_IN_ARRAY = -1.7
LARGE_TEST_VALUE_IN_ARRAY = 5.
LARGE_TEST_VALUE_NOT_IN_ARRAY = 4.6

MIN_GEQ_INDEX_SMALL_IN_ARRAY = 1
MIN_GEQ_INDEX_SMALL_NOT_IN_ARRAY = 2
MIN_GEQ_INDEX_LARGE_IN_ARRAY = 4
MIN_GEQ_INDEX_LARGE_NOT_IN_ARRAY = 4

MAX_LEQ_INDEX_SMALL_IN_ARRAY = 1
MAX_LEQ_INDEX_SMALL_NOT_IN_ARRAY = 1
MAX_LEQ_INDEX_LARGE_IN_ARRAY = 4
MAX_LEQ_INDEX_LARGE_NOT_IN_ARRAY = 3


class GriddedForecastsTests(unittest.TestCase):
    """Each method is a unit test for gridded_forecasts.py."""

    def test_column_name_to_distance_buffer_small_buffer_latlng(self):
        """Ensures correct output from _column_name_to_distance_buffer.

        In this case, the buffer is "inside storm" and the column name is for
        lat-long polygons.
        """

        this_min_distance_metres, this_max_distance_metres = (
            gridded_forecasts._column_name_to_distance_buffer(
                SMALL_BUFFER_LATLNG_COLUMN))

        self.assertTrue(numpy.isclose(
            this_min_distance_metres, SMALL_BUFFER_MIN_DISTANCE_METRES,
            equal_nan=True, atol=TOLERANCE))
        self.assertTrue(numpy.isclose(
            this_max_distance_metres, SMALL_BUFFER_MAX_DISTANCE_METRES,
            atol=TOLERANCE))

    def test_column_name_to_distance_buffer_small_buffer_xy(self):
        """Ensures correct output from _column_name_to_distance_buffer.

        In this case, the buffer is "inside storm" and the column name is for
        x-y polygons.
        """

        this_min_distance_metres, this_max_distance_metres = (
            gridded_forecasts._column_name_to_distance_buffer(
                SMALL_BUFFER_XY_COLUMN))

        self.assertTrue(numpy.isclose(
            this_min_distance_metres, SMALL_BUFFER_MIN_DISTANCE_METRES,
            equal_nan=True, atol=TOLERANCE))
        self.assertTrue(numpy.isclose(
            this_max_distance_metres, SMALL_BUFFER_MAX_DISTANCE_METRES,
            atol=TOLERANCE))

    def test_column_name_to_distance_buffer_small_buffer_forecast(self):
        """Ensures correct output from _column_name_to_distance_buffer.

        In this case, the buffer is "inside storm" and the column name is for
        forecast probabilities.
        """

        this_min_distance_metres, this_max_distance_metres = (
            gridded_forecasts._column_name_to_distance_buffer(
                SMALL_BUFFER_FORECAST_COLUMN))

        self.assertTrue(numpy.isclose(
            this_min_distance_metres, SMALL_BUFFER_MIN_DISTANCE_METRES,
            equal_nan=True, atol=TOLERANCE))
        self.assertTrue(numpy.isclose(
            this_max_distance_metres, SMALL_BUFFER_MAX_DISTANCE_METRES,
            atol=TOLERANCE))

    def test_column_name_to_distance_buffer_small_buffer_grid_rows(self):
        """Ensures correct output from _column_name_to_distance_buffer.

        In this case, the buffer is "inside storm" and the column name is for
        grid rows inside the polygon.
        """

        this_min_distance_metres, this_max_distance_metres = (
            gridded_forecasts._column_name_to_distance_buffer(
                SMALL_BUFFER_GRID_ROWS_COLUMN))

        self.assertTrue(numpy.isclose(
            this_min_distance_metres, SMALL_BUFFER_MIN_DISTANCE_METRES,
            equal_nan=True, atol=TOLERANCE))
        self.assertTrue(numpy.isclose(
            this_max_distance_metres, SMALL_BUFFER_MAX_DISTANCE_METRES,
            atol=TOLERANCE))

    def test_column_name_to_distance_buffer_small_buffer_grid_columns(self):
        """Ensures correct output from _column_name_to_distance_buffer.

        In this case, the buffer is "inside storm" and the column name is for
        grid columns inside the polygon.
        """

        this_min_distance_metres, this_max_distance_metres = (
            gridded_forecasts._column_name_to_distance_buffer(
                SMALL_BUFFER_GRID_COLUMNS_COLUMN))

        self.assertTrue(numpy.isclose(
            this_min_distance_metres, SMALL_BUFFER_MIN_DISTANCE_METRES,
            equal_nan=True, atol=TOLERANCE))
        self.assertTrue(numpy.isclose(
            this_max_distance_metres, SMALL_BUFFER_MAX_DISTANCE_METRES,
            atol=TOLERANCE))

    def test_column_name_to_distance_buffer_large_buffer_latlng(self):
        """Ensures correct output from _column_name_to_distance_buffer.

        In this case, the buffer is "5-10 km outside storm" and the column name
        is for lat-long polygons.
        """

        this_min_distance_metres, this_max_distance_metres = (
            gridded_forecasts._column_name_to_distance_buffer(
                LARGE_BUFFER_LATLNG_COLUMN))

        self.assertTrue(numpy.isclose(
            this_min_distance_metres, LARGE_BUFFER_MIN_DISTANCE_METRES,
            atol=TOLERANCE))
        self.assertTrue(numpy.isclose(
            this_max_distance_metres, LARGE_BUFFER_MAX_DISTANCE_METRES,
            atol=TOLERANCE))

    def test_column_name_to_distance_buffer_large_buffer_xy(self):
        """Ensures correct output from _column_name_to_distance_buffer.

        In this case, the buffer is "5-10 km outside storm" and the column name
        is for x-y polygons.
        """

        this_min_distance_metres, this_max_distance_metres = (
            gridded_forecasts._column_name_to_distance_buffer(
                LARGE_BUFFER_XY_COLUMN))

        self.assertTrue(numpy.isclose(
            this_min_distance_metres, LARGE_BUFFER_MIN_DISTANCE_METRES,
            atol=TOLERANCE))
        self.assertTrue(numpy.isclose(
            this_max_distance_metres, LARGE_BUFFER_MAX_DISTANCE_METRES,
            atol=TOLERANCE))

    def test_column_name_to_distance_buffer_large_buffer_forecast(self):
        """Ensures correct output from _column_name_to_distance_buffer.

        In this case, the buffer is "5-10 km outside storm" and the column name
        is for forecast probabilities.
        """

        this_min_distance_metres, this_max_distance_metres = (
            gridded_forecasts._column_name_to_distance_buffer(
                LARGE_BUFFER_FORECAST_COLUMN))

        self.assertTrue(numpy.isclose(
            this_min_distance_metres, LARGE_BUFFER_MIN_DISTANCE_METRES,
            atol=TOLERANCE))
        self.assertTrue(numpy.isclose(
            this_max_distance_metres, LARGE_BUFFER_MAX_DISTANCE_METRES,
            atol=TOLERANCE))

    def test_column_name_to_distance_buffer_large_buffer_grid_rows(self):
        """Ensures correct output from _column_name_to_distance_buffer.

        In this case, the buffer is "5-10 km outside storm" and the column name
        is for grid rows inside the polygon.
        """

        this_min_distance_metres, this_max_distance_metres = (
            gridded_forecasts._column_name_to_distance_buffer(
                LARGE_BUFFER_GRID_ROWS_COLUMN))

        self.assertTrue(numpy.isclose(
            this_min_distance_metres, LARGE_BUFFER_MIN_DISTANCE_METRES,
            atol=TOLERANCE))
        self.assertTrue(numpy.isclose(
            this_max_distance_metres, LARGE_BUFFER_MAX_DISTANCE_METRES,
            atol=TOLERANCE))

    def test_column_name_to_distance_buffer_large_buffer_grid_columns(self):
        """Ensures correct output from _column_name_to_distance_buffer.

        In this case, the buffer is "5-10 km outside storm" and the column name
        is for grid columns inside the polygon.
        """

        this_min_distance_metres, this_max_distance_metres = (
            gridded_forecasts._column_name_to_distance_buffer(
                LARGE_BUFFER_GRID_COLUMNS_COLUMN))

        self.assertTrue(numpy.isclose(
            this_min_distance_metres, LARGE_BUFFER_MIN_DISTANCE_METRES,
            atol=TOLERANCE))
        self.assertTrue(numpy.isclose(
            this_max_distance_metres, LARGE_BUFFER_MAX_DISTANCE_METRES,
            atol=TOLERANCE))

    def test_distance_buffer_to_column_name_small_buffer_latlng(self):
        """Ensures correct output from _distance_buffer_to_column_name.

        In this case, the buffer is "inside storm" and the column name is for
        lat-long polygons.
        """

        this_column_name = gridded_forecasts._distance_buffer_to_column_name(
            SMALL_BUFFER_MIN_DISTANCE_METRES, SMALL_BUFFER_MAX_DISTANCE_METRES,
            column_type=gridded_forecasts.LATLNG_POLYGON_COLUMN_TYPE)
        self.assertTrue(this_column_name == SMALL_BUFFER_LATLNG_COLUMN)

    def test_distance_buffer_to_column_name_small_buffer_xy(self):
        """Ensures correct output from _distance_buffer_to_column_name.

        In this case, the buffer is "inside storm" and the column name is for
        x-y polygons.
        """

        this_column_name = gridded_forecasts._distance_buffer_to_column_name(
            SMALL_BUFFER_MIN_DISTANCE_METRES, SMALL_BUFFER_MAX_DISTANCE_METRES,
            column_type=gridded_forecasts.XY_POLYGON_COLUMN_TYPE)
        self.assertTrue(this_column_name == SMALL_BUFFER_XY_COLUMN)

    def test_distance_buffer_to_column_name_small_buffer_forecast(self):
        """Ensures correct output from _distance_buffer_to_column_name.

        In this case, the buffer is "inside storm" and the column name is for
        forecast probabilities.
        """

        this_column_name = gridded_forecasts._distance_buffer_to_column_name(
            SMALL_BUFFER_MIN_DISTANCE_METRES, SMALL_BUFFER_MAX_DISTANCE_METRES,
            column_type=gridded_forecasts.FORECAST_COLUMN_TYPE)
        self.assertTrue(this_column_name == SMALL_BUFFER_FORECAST_COLUMN)

    def test_distance_buffer_to_column_name_small_buffer_grid_rows(self):
        """Ensures correct output from _distance_buffer_to_column_name.

        In this case, the buffer is "inside storm" and the column name is for
        grid rows inside the polygon.
        """

        this_column_name = gridded_forecasts._distance_buffer_to_column_name(
            SMALL_BUFFER_MIN_DISTANCE_METRES, SMALL_BUFFER_MAX_DISTANCE_METRES,
            column_type=gridded_forecasts.GRID_ROWS_IN_POLYGON_COLUMN_TYPE)
        self.assertTrue(this_column_name == SMALL_BUFFER_GRID_ROWS_COLUMN)

    def test_distance_buffer_to_column_name_small_buffer_grid_columns(self):
        """Ensures correct output from _distance_buffer_to_column_name.

        In this case, the buffer is "inside storm" and the column name is for
        grid columns inside the polygon.
        """

        this_column_name = gridded_forecasts._distance_buffer_to_column_name(
            SMALL_BUFFER_MIN_DISTANCE_METRES, SMALL_BUFFER_MAX_DISTANCE_METRES,
            column_type=gridded_forecasts.GRID_COLUMNS_IN_POLYGON_COLUMN_TYPE)
        self.assertTrue(this_column_name == SMALL_BUFFER_GRID_COLUMNS_COLUMN)

    def test_distance_buffer_to_column_name_large_buffer_latlng(self):
        """Ensures correct output from _distance_buffer_to_column_name.

        In this case, the buffer is "5-10 km outside storm" and the column name
        is for lat-long polygons.
        """

        this_column_name = gridded_forecasts._distance_buffer_to_column_name(
            LARGE_BUFFER_MIN_DISTANCE_METRES, LARGE_BUFFER_MAX_DISTANCE_METRES,
            column_type=gridded_forecasts.LATLNG_POLYGON_COLUMN_TYPE)
        self.assertTrue(this_column_name == LARGE_BUFFER_LATLNG_COLUMN)

    def test_distance_buffer_to_column_name_large_buffer_xy(self):
        """Ensures correct output from _distance_buffer_to_column_name.

        In this case, the buffer is "5-10 km outside storm" and the column name
        is for x-y polygons.
        """

        this_column_name = gridded_forecasts._distance_buffer_to_column_name(
            LARGE_BUFFER_MIN_DISTANCE_METRES, LARGE_BUFFER_MAX_DISTANCE_METRES,
            column_type=gridded_forecasts.XY_POLYGON_COLUMN_TYPE)
        self.assertTrue(this_column_name == LARGE_BUFFER_XY_COLUMN)

    def test_distance_buffer_to_column_name_large_buffer_forecast(self):
        """Ensures correct output from _distance_buffer_to_column_name.

        In this case, the buffer is "5-10 km outside storm" and the column name
        is for forecast probabilities.
        """

        this_column_name = gridded_forecasts._distance_buffer_to_column_name(
            LARGE_BUFFER_MIN_DISTANCE_METRES, LARGE_BUFFER_MAX_DISTANCE_METRES,
            column_type=gridded_forecasts.FORECAST_COLUMN_TYPE)
        self.assertTrue(this_column_name == LARGE_BUFFER_FORECAST_COLUMN)

    def test_distance_buffer_to_column_name_large_buffer_grid_rows(self):
        """Ensures correct output from _distance_buffer_to_column_name.

        In this case, the buffer is "5-10 km outside storm" and the column name
        is for grid rows inside the polygon.
        """

        this_column_name = gridded_forecasts._distance_buffer_to_column_name(
            LARGE_BUFFER_MIN_DISTANCE_METRES, LARGE_BUFFER_MAX_DISTANCE_METRES,
            column_type=gridded_forecasts.GRID_ROWS_IN_POLYGON_COLUMN_TYPE)
        self.assertTrue(this_column_name == LARGE_BUFFER_GRID_ROWS_COLUMN)

    def test_distance_buffer_to_column_name_large_buffer_grid_columns(self):
        """Ensures correct output from _distance_buffer_to_column_name.

        In this case, the buffer is "5-10 km outside storm" and the column name
        is for grid columns inside the polygon.
        """

        this_column_name = gridded_forecasts._distance_buffer_to_column_name(
            LARGE_BUFFER_MIN_DISTANCE_METRES, LARGE_BUFFER_MAX_DISTANCE_METRES,
            column_type=gridded_forecasts.GRID_COLUMNS_IN_POLYGON_COLUMN_TYPE)
        self.assertTrue(this_column_name == LARGE_BUFFER_GRID_COLUMNS_COLUMN)

    def test_get_distance_buffer_columns_latlng(self):
        """Ensures correct output from _get_distance_buffer_columns.

        In this case, should return columns with lat-long polygons only.
        """

        these_column_names = gridded_forecasts._get_distance_buffer_columns(
            EMPTY_STORM_OBJECT_TABLE,
            column_type=gridded_forecasts.LATLNG_POLYGON_COLUMN_TYPE)
        self.assertTrue(
            set(these_column_names) ==
            {SMALL_BUFFER_LATLNG_COLUMN, LARGE_BUFFER_LATLNG_COLUMN})

    def test_get_distance_buffer_columns_xy(self):
        """Ensures correct output from _get_distance_buffer_columns.

        In this case, should return columns with x-y polygons only.
        """

        these_column_names = gridded_forecasts._get_distance_buffer_columns(
            EMPTY_STORM_OBJECT_TABLE,
            column_type=gridded_forecasts.XY_POLYGON_COLUMN_TYPE)
        self.assertTrue(
            set(these_column_names) ==
            {SMALL_BUFFER_XY_COLUMN, LARGE_BUFFER_XY_COLUMN})

    def test_get_distance_buffer_columns_forecast(self):
        """Ensures correct output from _get_distance_buffer_columns.

        In this case, should return columns with forecast probabilities only.
        """

        these_column_names = gridded_forecasts._get_distance_buffer_columns(
            EMPTY_STORM_OBJECT_TABLE,
            column_type=gridded_forecasts.FORECAST_COLUMN_TYPE)
        self.assertTrue(
            set(these_column_names) ==
            {SMALL_BUFFER_FORECAST_COLUMN, LARGE_BUFFER_FORECAST_COLUMN})

    def test_get_distance_buffer_columns_grid_rows(self):
        """Ensures correct output from _get_distance_buffer_columns.

        In this case, should return columns with grid rows only.
        """

        these_column_names = gridded_forecasts._get_distance_buffer_columns(
            EMPTY_STORM_OBJECT_TABLE,
            column_type=gridded_forecasts.GRID_ROWS_IN_POLYGON_COLUMN_TYPE)
        self.assertTrue(
            set(these_column_names) ==
            {SMALL_BUFFER_GRID_ROWS_COLUMN, LARGE_BUFFER_GRID_ROWS_COLUMN})

    def test_get_distance_buffer_columns_grid_columns(self):
        """Ensures correct output from _get_distance_buffer_columns.

        In this case, should return columns with grid columns only.
        """

        these_column_names = gridded_forecasts._get_distance_buffer_columns(
            EMPTY_STORM_OBJECT_TABLE,
            column_type=gridded_forecasts.GRID_COLUMNS_IN_POLYGON_COLUMN_TYPE)
        self.assertTrue(set(these_column_names) ==
                        {SMALL_BUFFER_GRID_COLUMNS_COLUMN,
                         LARGE_BUFFER_GRID_COLUMNS_COLUMN})

    def test_check_distance_buffers_all_good(self):
        """Ensures correct output from _check_distance_buffers.

        In this case the set of distance buffers is valid.
        """

        min_distances_metres = numpy.array(
            [SMALL_BUFFER_MIN_DISTANCE_METRES,
             MEDIUM_BUFFER_MIN_DISTANCE_METRES,
             LARGE_BUFFER_MIN_DISTANCE_METRES])
        max_distances_metres = numpy.array(
            [SMALL_BUFFER_MAX_DISTANCE_METRES,
             MEDIUM_BUFFER_MAX_DISTANCE_METRES,
             LARGE_BUFFER_MAX_DISTANCE_METRES])

        gridded_forecasts._check_distance_buffers(
            min_distances_metres, max_distances_metres)

    def test_check_distance_buffers_non_unique(self):
        """Ensures correct output from _check_distance_buffers.

        In this case, distance buffers are non-unique.
        """

        min_distances_metres = numpy.array(
            [SMALL_BUFFER_MIN_DISTANCE_METRES, SMALL_BUFFER_MIN_DISTANCE_METRES,
             MEDIUM_BUFFER_MIN_DISTANCE_METRES,
             LARGE_BUFFER_MIN_DISTANCE_METRES])
        max_distances_metres = numpy.array(
            [SMALL_BUFFER_MAX_DISTANCE_METRES, SMALL_BUFFER_MAX_DISTANCE_METRES,
             MEDIUM_BUFFER_MAX_DISTANCE_METRES,
             LARGE_BUFFER_MAX_DISTANCE_METRES])

        with self.assertRaises(ValueError):
            gridded_forecasts._check_distance_buffers(
                min_distances_metres, max_distances_metres)

    def test_check_distance_buffers_non_abutting_4999m(self):
        """Ensures correct output from _check_distance_buffers.

        In this case, distance buffers are non-abutting at 4999 metres.
        """

        min_distances_metres = numpy.array(
            [SMALL_BUFFER_MIN_DISTANCE_METRES,
             MEDIUM_BUFFER_MIN_DISTANCE_METRES,
             LARGE_BUFFER_MIN_DISTANCE_METRES])
        max_distances_metres = numpy.array(
            [SMALL_BUFFER_MAX_DISTANCE_METRES,
             MEDIUM_BUFFER_MAX_DISTANCE_METRES - 1.,
             LARGE_BUFFER_MAX_DISTANCE_METRES])

        with self.assertRaises(ValueError):
            gridded_forecasts._check_distance_buffers(
                min_distances_metres, max_distances_metres)

    def test_check_distance_buffers_non_abutting_5001m(self):
        """Ensures correct output from _check_distance_buffers.

        In this case, distance buffers are non-abutting at 5001 metres.
        """

        min_distances_metres = numpy.array(
            [SMALL_BUFFER_MIN_DISTANCE_METRES,
             MEDIUM_BUFFER_MIN_DISTANCE_METRES,
             LARGE_BUFFER_MIN_DISTANCE_METRES + 1.])
        max_distances_metres = numpy.array(
            [SMALL_BUFFER_MAX_DISTANCE_METRES,
             MEDIUM_BUFFER_MAX_DISTANCE_METRES,
             LARGE_BUFFER_MAX_DISTANCE_METRES])

        with self.assertRaises(ValueError):
            gridded_forecasts._check_distance_buffers(
                min_distances_metres, max_distances_metres)

    def test_create_xy_grid(self):
        """Ensures correct output from _create_xy_grid."""

        these_x_metres, these_y_metres = gridded_forecasts._create_xy_grid(
            STORM_OBJECT_TABLE, x_spacing_metres=GRID_SPACING_X_METRES,
            y_spacing_metres=GRID_SPACING_Y_METRES,
            max_lead_time_sec=MAX_LEAD_TIME_SEC)

        self.assertTrue(numpy.allclose(
            these_x_metres, GRID_POINTS_X_METRES, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_y_metres, GRID_POINTS_Y_METRES, atol=TOLERANCE))

    def test_create_latlng_grid(self):
        """Ensures correct output from _create_latlng_grid."""

        these_latitudes_deg, these_longitudes_deg = (
            gridded_forecasts._create_latlng_grid(
                grid_points_x_metres=GRID_POINTS_X_METRES,
                grid_points_y_metres=GRID_POINTS_Y_METRES,
                projection_object=PROJECTION_OBJECT,
                latitude_spacing_deg=GRID_LAT_SPACING_DEG,
                longitude_spacing_deg=GRID_LNG_SPACING_DEG))

        self.assertTrue(numpy.allclose(
            these_latitudes_deg, GRID_POINT_LATITUDES_DEG, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_longitudes_deg, GRID_POINT_LONGITUDES_DEG, atol=TOLERANCE))

    def test_normalize_probs_by_polygon_area(self):
        """Ensures correct output from _normalize_probs_by_polygon_area."""

        orig_storm_object_table = copy.deepcopy(
            STORM_OBJECT_TABLE)
        new_storm_object_table = (
            gridded_forecasts._normalize_probs_by_polygon_area(
                orig_storm_object_table, PROB_RADIUS_FOR_GRID_METRES))

        self.assertTrue(numpy.allclose(
            new_storm_object_table[SMALL_BUFFER_FORECAST_COLUMN].values,
            SMALL_BUFFER_NORMALIZED_PROBS, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            new_storm_object_table[LARGE_BUFFER_FORECAST_COLUMN].values,
            LARGE_BUFFER_NORMALIZED_PROBS, atol=TOLERANCE))

    def test_find_grid_points_in_polygon_small_buffer(self):
        """Ensures correct output from _find_grid_points_in_polygon.

        In this case, input polygon is for small distance buffer.
        """

        these_rows, these_columns = (
            gridded_forecasts._find_grid_points_in_polygon(
                SMALL_BUFFER_POLYGON_OBJECT_XY,
                grid_points_x_metres=GRID_POINTS_FOR_PIP_X_METRES,
                grid_points_y_metres=GRID_POINTS_FOR_PIP_Y_METRES))

        self.assertTrue(numpy.array_equal(
            these_rows, GRID_ROWS_IN_SMALL_BUFFER))
        self.assertTrue(numpy.array_equal(
            these_columns, GRID_COLUMNS_IN_SMALL_BUFFER))

    def test_find_grid_points_in_polygon_large_buffer(self):
        """Ensures correct output from _find_grid_points_in_polygon.

        In this case, input polygon is for large distance buffer.
        """

        these_rows, these_columns = (
            gridded_forecasts._find_grid_points_in_polygon(
                LARGE_BUFFER_POLYGON_OBJECT_XY,
                grid_points_x_metres=GRID_POINTS_FOR_PIP_X_METRES,
                grid_points_y_metres=GRID_POINTS_FOR_PIP_Y_METRES))

        self.assertTrue(numpy.array_equal(
            these_rows, GRID_ROWS_IN_LARGE_BUFFER))
        self.assertTrue(numpy.array_equal(
            these_columns, GRID_COLUMNS_IN_LARGE_BUFFER))

    def test_find_min_value_greater_or_equal_small_in_array(self):
        """Ensures correct output from _find_min_value_greater_or_equal.

        In this case, the test value is small and matches a member of the input
        array.
        """

        _, this_index = gridded_forecasts._find_min_value_greater_or_equal(
            SORTED_ARRAY, SMALL_TEST_VALUE_IN_ARRAY)
        self.assertTrue(this_index == MIN_GEQ_INDEX_SMALL_IN_ARRAY)

    def test_find_min_value_greater_or_equal_small_not_in_array(self):
        """Ensures correct output from _find_min_value_greater_or_equal.

        In this case, the test value is small and does not occur in the input
        array.
        """

        _, this_index = gridded_forecasts._find_min_value_greater_or_equal(
            SORTED_ARRAY, SMALL_TEST_VALUE_NOT_IN_ARRAY)
        self.assertTrue(this_index == MIN_GEQ_INDEX_SMALL_NOT_IN_ARRAY)

    def test_find_min_value_greater_or_equal_large_in_array(self):
        """Ensures correct output from _find_min_value_greater_or_equal.

        In this case, the test value is large and matches a member of the input
        array.
        """

        _, this_index = gridded_forecasts._find_min_value_greater_or_equal(
            SORTED_ARRAY, LARGE_TEST_VALUE_IN_ARRAY)
        self.assertTrue(this_index == MIN_GEQ_INDEX_LARGE_IN_ARRAY)

    def test_find_min_value_greater_or_equal_large_not_in_array(self):
        """Ensures correct output from _find_min_value_greater_or_equal.

        In this case, the test value is large and does not occur in the input
        array.
        """

        _, this_index = gridded_forecasts._find_min_value_greater_or_equal(
            SORTED_ARRAY, LARGE_TEST_VALUE_NOT_IN_ARRAY)
        self.assertTrue(this_index == MIN_GEQ_INDEX_LARGE_NOT_IN_ARRAY)

    def test_find_max_value_less_than_or_equal_small_in_array(self):
        """Ensures correct output from _find_max_value_less_than_or_equal.

        In this case, the test value is small and matches a member of the input
        array.
        """

        _, this_index = gridded_forecasts._find_max_value_less_than_or_equal(
            SORTED_ARRAY, SMALL_TEST_VALUE_IN_ARRAY)
        self.assertTrue(this_index == MAX_LEQ_INDEX_SMALL_IN_ARRAY)

    def test_find_max_value_less_than_or_equal_small_not_in_array(self):
        """Ensures correct output from _find_max_value_less_than_or_equal.

        In this case, the test value is small and does not occur in the input
        array.
        """

        _, this_index = gridded_forecasts._find_max_value_less_than_or_equal(
            SORTED_ARRAY, SMALL_TEST_VALUE_NOT_IN_ARRAY)
        self.assertTrue(this_index == MAX_LEQ_INDEX_SMALL_NOT_IN_ARRAY)

    def test_find_max_value_less_than_or_equal_large_in_array(self):
        """Ensures correct output from _find_max_value_less_than_or_equal.

        In this case, the test value is large and matches a member of the input
        array.
        """

        _, this_index = gridded_forecasts._find_max_value_less_than_or_equal(
            SORTED_ARRAY, LARGE_TEST_VALUE_IN_ARRAY)
        self.assertTrue(this_index == MAX_LEQ_INDEX_LARGE_IN_ARRAY)

    def test_find_max_value_less_than_or_equal_large_not_in_array(self):
        """Ensures correct output from _find_max_value_less_than_or_equal.

        In this case, the test value is large and does not occur in the input
        array.
        """

        _, this_index = gridded_forecasts._find_max_value_less_than_or_equal(
            SORTED_ARRAY, LARGE_TEST_VALUE_NOT_IN_ARRAY)
        self.assertTrue(this_index == MAX_LEQ_INDEX_LARGE_NOT_IN_ARRAY)


if __name__ == '__main__':
    unittest.main()
