"""Unit tests for storm_tracking_utils.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import geodetic_utils

FAKE_DATA_SOURCE = 'foo'

MIN_BUFFER_DISTANCE_METRES = 0.
MAX_BUFFER_DISTANCE_METRES = 5000.
BUFFER_COLUMN_NAME_INCLUSIVE = 'polygon_object_latlng_buffer_5000m'
BUFFER_COLUMN_NAME_EXCLUSIVE = 'polygon_object_latlng_buffer_0m_5000m'
BUFFER_COLUMN_NAME_FAKE = 'poop'

# The following constants are used to test remove_rows_with_nan.
ARRAY_WITHOUT_NAN = numpy.array([1, 2, 3, 4, 5])
ARRAY_WITH_NAN = numpy.array([1, numpy.nan, 3, numpy.nan, 5])

THIS_DICTIONARY = {'without_nan': ARRAY_WITHOUT_NAN,
                   'with_nan': ARRAY_WITH_NAN}
TABLE_WITH_NAN_ROWS = pandas.DataFrame.from_dict(THIS_DICTIONARY)

ROWS_WITH_NAN = numpy.array([1, 3], dtype=int)
TABLE_WITHOUT_NAN_ROWS = TABLE_WITH_NAN_ROWS.drop(
    TABLE_WITH_NAN_ROWS.index[ROWS_WITH_NAN], axis=0, inplace=False)

# The following constants are used to test _get_grid_points_in_storms.
NW_GRID_POINT_LAT_DEG = 53.5
NW_GRID_POINT_LNG_DEG = 246.5
LATITUDE_SPACING_DEG = 0.01
LONGITUDE_SPACING_DEG = 0.01
NUM_GRID_ROWS = 7
NUM_GRID_COLUMNS = 8

NUM_STORMS_SMALL_SCALE = 3
STORM_IDS_SMALL_SCALE = ['i', 'ii', 'iii']
GRID_POINT_ROWS_STORM_I = numpy.array([1, 1, 2, 2], dtype=int)
GRID_POINT_COLUMNS_STORM_I = numpy.array([1, 2, 1, 2], dtype=int)
GRID_POINT_LAT_STORM_I_DEG = numpy.array([53.49, 53.48, 53.49, 53.48])
GRID_POINT_LNG_STORM_I_DEG = numpy.array([246.51, 246.52, 246.51, 246.52])

GRID_POINT_ROWS_STORM_II = numpy.array([1, 1, 2, 2], dtype=int)
GRID_POINT_COLUMNS_STORM_II = numpy.array([4, 5, 4, 5], dtype=int)
GRID_POINT_LAT_STORM_II_DEG = numpy.array([53.49, 53.48, 53.49, 53.48])
GRID_POINT_LNG_STORM_II_DEG = numpy.array([246.54, 246.55, 246.54, 246.55])

GRID_POINT_ROWS_STORM_III = numpy.array([4, 4, 5, 5], dtype=int)
GRID_POINT_COLUMNS_STORM_III = numpy.array([4, 5, 4, 5], dtype=int)
GRID_POINT_LAT_STORM_III_DEG = numpy.array([53.46, 53.45, 53.46, 53.45])
GRID_POINT_LNG_STORM_III_DEG = numpy.array([246.54, 246.55, 246.54, 246.55])

GRID_POINT_ROWS_BY_STORM = [
    GRID_POINT_ROWS_STORM_I, GRID_POINT_ROWS_STORM_II,
    GRID_POINT_ROWS_STORM_III]
GRID_POINT_COLUMNS_BY_STORM = [
    GRID_POINT_COLUMNS_STORM_I, GRID_POINT_COLUMNS_STORM_II,
    GRID_POINT_COLUMNS_STORM_III]
GRID_POINT_LAT_BY_STORM_DEG = [
    GRID_POINT_LAT_STORM_I_DEG, GRID_POINT_LAT_STORM_II_DEG,
    GRID_POINT_LAT_STORM_III_DEG]
GRID_POINT_LNG_BY_STORM_DEG = [
    GRID_POINT_LNG_STORM_I_DEG, GRID_POINT_LNG_STORM_II_DEG,
    GRID_POINT_LNG_STORM_III_DEG]

STORM_OBJECT_DICT_SMALL_SCALE = {
    tracking_utils.STORM_ID_COLUMN: STORM_IDS_SMALL_SCALE}
STORM_OBJECT_TABLE_SMALL_SCALE = pandas.DataFrame.from_dict(
    STORM_OBJECT_DICT_SMALL_SCALE)

SIMPLE_ARRAY = numpy.full(NUM_STORMS_SMALL_SCALE, numpy.nan)
OBJECT_ARRAY = numpy.full(NUM_STORMS_SMALL_SCALE, numpy.nan, dtype=object)
NESTED_ARRAY = STORM_OBJECT_TABLE_SMALL_SCALE[[
    tracking_utils.STORM_ID_COLUMN,
    tracking_utils.STORM_ID_COLUMN]].values.tolist()

ARGUMENT_DICT = {tracking_utils.CENTROID_LAT_COLUMN: SIMPLE_ARRAY,
                 tracking_utils.CENTROID_LNG_COLUMN: SIMPLE_ARRAY,
                 tracking_utils.GRID_POINT_LAT_COLUMN: NESTED_ARRAY,
                 tracking_utils.GRID_POINT_LNG_COLUMN: NESTED_ARRAY,
                 tracking_utils.GRID_POINT_ROW_COLUMN: NESTED_ARRAY,
                 tracking_utils.GRID_POINT_COLUMN_COLUMN: NESTED_ARRAY,
                 tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN: OBJECT_ARRAY,
                 tracking_utils.POLYGON_OBJECT_ROWCOL_COLUMN: OBJECT_ARRAY}
STORM_OBJECT_TABLE_SMALL_SCALE = STORM_OBJECT_TABLE_SMALL_SCALE.assign(
    **ARGUMENT_DICT)

for i in range(NUM_STORMS_SMALL_SCALE):
    STORM_OBJECT_TABLE_SMALL_SCALE[tracking_utils.GRID_POINT_LAT_COLUMN].values[
        i] = GRID_POINT_LAT_BY_STORM_DEG[i]
    STORM_OBJECT_TABLE_SMALL_SCALE[tracking_utils.GRID_POINT_LNG_COLUMN].values[
        i] = GRID_POINT_LNG_BY_STORM_DEG[i]
    STORM_OBJECT_TABLE_SMALL_SCALE[tracking_utils.GRID_POINT_ROW_COLUMN].values[
        i] = GRID_POINT_ROWS_BY_STORM[i]
    STORM_OBJECT_TABLE_SMALL_SCALE[
        tracking_utils.GRID_POINT_COLUMN_COLUMN
    ].values[i] = GRID_POINT_COLUMNS_BY_STORM[i]

    THESE_VERTEX_ROWS, THESE_VERTEX_COLUMNS = (
        polygons.grid_points_in_poly_to_vertices(
            GRID_POINT_ROWS_BY_STORM[i], GRID_POINT_COLUMNS_BY_STORM[i]))

    THESE_VERTEX_LATITUDES_DEG, THESE_VERTEX_LONGITUDES_DEG = (
        radar_utils.rowcol_to_latlng(
            THESE_VERTEX_ROWS, THESE_VERTEX_COLUMNS,
            nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
            nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
            lat_spacing_deg=LATITUDE_SPACING_DEG,
            lng_spacing_deg=LONGITUDE_SPACING_DEG))

    (THIS_CENTROID_LAT_DEG, THIS_CENTROID_LNG_DEG
    ) = geodetic_utils.get_latlng_centroid(
        latitudes_deg=THESE_VERTEX_LATITUDES_DEG,
        longitudes_deg=THESE_VERTEX_LONGITUDES_DEG)

    STORM_OBJECT_TABLE_SMALL_SCALE[tracking_utils.CENTROID_LAT_COLUMN].values[
        i] = THIS_CENTROID_LAT_DEG
    STORM_OBJECT_TABLE_SMALL_SCALE[tracking_utils.CENTROID_LNG_COLUMN].values[
        i] = THIS_CENTROID_LNG_DEG

    STORM_OBJECT_TABLE_SMALL_SCALE[
        tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN].values[i] = (
            polygons.vertex_arrays_to_polygon_object(
                THESE_VERTEX_LONGITUDES_DEG, THESE_VERTEX_LATITUDES_DEG))
    STORM_OBJECT_TABLE_SMALL_SCALE[
        tracking_utils.POLYGON_OBJECT_ROWCOL_COLUMN].values[i] = (
            polygons.vertex_arrays_to_polygon_object(
                THESE_VERTEX_COLUMNS, THESE_VERTEX_ROWS))

FLAT_GRID_POINT_INDICES_STORM_I = numpy.array([9, 10, 17, 18], dtype=int)
FLAT_GRID_POINT_INDICES_STORM_II = numpy.array([12, 13, 20, 21], dtype=int)
FLAT_GRID_POINT_INDICES_STORM_III = numpy.array([36, 37, 44, 45], dtype=int)
FLATTENED_GRID_POINT_INDICES = numpy.concatenate((
    FLAT_GRID_POINT_INDICES_STORM_I, FLAT_GRID_POINT_INDICES_STORM_II,
    FLAT_GRID_POINT_INDICES_STORM_III))
FLATTENED_STORM_IDS = [
    'i', 'i', 'i', 'i', 'ii', 'ii', 'ii', 'ii', 'iii', 'iii', 'iii', 'iii']

GRID_POINTS_IN_STORMS_DICT = {
    tracking_utils.FLATTENED_INDEX_COLUMN: FLATTENED_GRID_POINT_INDICES,
    tracking_utils.STORM_ID_COLUMN: FLATTENED_STORM_IDS
}
GRID_POINTS_IN_STORMS_TABLE = pandas.DataFrame.from_dict(
    GRID_POINTS_IN_STORMS_DICT)

# The following constants are used to test merge_storms_at_two_scales.
NUM_STORMS_LARGE_SCALE = 3
STORM_IDS_LARGE_SCALE = ['a', 'b', 'c']
GRID_POINT_ROWS_STORM_A = numpy.array(
    [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6], dtype=int)
GRID_POINT_COLUMNS_STORM_A = numpy.array(
    [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=int)
GRID_POINT_LAT_STORM_A_DEG = numpy.array(
    [53.53, 53.53, 53.53, 53.53, 53.54, 53.54, 53.54, 53.54, 53.55, 53.55,
     53.55, 53.55, 53.56, 53.56, 53.56, 53.56])
GRID_POINT_LNG_STORM_A_DEG = numpy.array(
    [246.5, 246.51, 246.52, 246.53, 246.5, 246.51, 246.52, 246.53, 246.5,
     246.51, 246.52, 246.53, 246.5, 246.51, 246.52, 246.53])

GRID_POINT_ROWS_STORM_B = numpy.array(
    [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6], dtype=int)
GRID_POINT_COLUMNS_STORM_B = numpy.array(
    [4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7], dtype=int)
GRID_POINT_LAT_STORM_B_DEG = numpy.array(
    [53.53, 53.53, 53.53, 53.53, 53.54, 53.54, 53.54, 53.54, 53.55, 53.55,
     53.55, 53.55, 53.56, 53.56, 53.56, 53.56])
GRID_POINT_LNG_STORM_B_DEG = numpy.array(
    [246.54, 246.55, 246.56, 246.57, 246.54, 246.55, 246.56, 246.57, 246.54,
     246.55, 246.56, 246.57, 246.54, 246.55, 246.56, 246.57])

GRID_POINT_ROWS_STORM_C = numpy.array(
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
     6, 6, 6, 6], dtype=int)
GRID_POINT_COLUMNS_STORM_C = numpy.array(
    [4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
     4, 5, 6, 7], dtype=int)
GRID_POINT_LAT_STORM_C_DEG = numpy.array(
    [53.5, 53.5, 53.5, 53.5, 53.51, 53.51, 53.51, 53.51,
     53.52, 53.52, 53.52, 53.52, 53.53, 53.53, 53.53, 53.53,
     53.54, 53.54, 53.54, 53.54, 53.55, 53.55, 53.55, 53.55,
     53.56, 53.56, 53.56, 53.56])
GRID_POINT_LNG_STORM_C_DEG = numpy.array(
    [246.54, 246.55, 246.56, 246.57, 246.54, 246.55, 246.56, 246.57,
     246.54, 246.55, 246.56, 246.57, 246.54, 246.55, 246.56, 246.57,
     246.54, 246.55, 246.56, 246.57, 246.54, 246.55, 246.56, 246.57,
     246.54, 246.55, 246.56, 246.57])

GRID_POINT_ROWS_BY_STORM = [
    GRID_POINT_ROWS_STORM_A, GRID_POINT_ROWS_STORM_B, GRID_POINT_ROWS_STORM_C]
GRID_POINT_COLUMNS_BY_STORM = [
    GRID_POINT_COLUMNS_STORM_A, GRID_POINT_COLUMNS_STORM_B,
    GRID_POINT_COLUMNS_STORM_C]
GRID_POINT_LAT_BY_STORM_DEG = [
    GRID_POINT_LAT_STORM_A_DEG, GRID_POINT_LAT_STORM_B_DEG,
    GRID_POINT_LAT_STORM_C_DEG]
GRID_POINT_LNG_BY_STORM_DEG = [
    GRID_POINT_LNG_STORM_A_DEG, GRID_POINT_LNG_STORM_B_DEG,
    GRID_POINT_LNG_STORM_C_DEG]

STORM_OBJECT_TABLE_LARGE_SCALE = copy.deepcopy(STORM_OBJECT_TABLE_SMALL_SCALE)
ARGUMENT_DICT = {tracking_utils.STORM_ID_COLUMN: STORM_IDS_LARGE_SCALE}
STORM_OBJECT_TABLE_LARGE_SCALE.assign(**ARGUMENT_DICT)

for i in range(NUM_STORMS_LARGE_SCALE):
    STORM_OBJECT_TABLE_LARGE_SCALE[tracking_utils.GRID_POINT_LAT_COLUMN].values[
        i] = GRID_POINT_LAT_BY_STORM_DEG[i]
    STORM_OBJECT_TABLE_LARGE_SCALE[tracking_utils.GRID_POINT_LNG_COLUMN].values[
        i] = GRID_POINT_LNG_BY_STORM_DEG[i]
    STORM_OBJECT_TABLE_LARGE_SCALE[tracking_utils.GRID_POINT_ROW_COLUMN].values[
        i] = GRID_POINT_ROWS_BY_STORM[i]
    STORM_OBJECT_TABLE_LARGE_SCALE[
        tracking_utils.GRID_POINT_COLUMN_COLUMN
    ].values[i] = GRID_POINT_COLUMNS_BY_STORM[i]

    THESE_VERTEX_ROWS, THESE_VERTEX_COLUMNS = (
        polygons.grid_points_in_poly_to_vertices(
            GRID_POINT_ROWS_BY_STORM[i], GRID_POINT_COLUMNS_BY_STORM[i]))

    THESE_VERTEX_LATITUDES_DEG, THESE_VERTEX_LONGITUDES_DEG = (
        radar_utils.rowcol_to_latlng(
            THESE_VERTEX_ROWS, THESE_VERTEX_COLUMNS,
            nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
            nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
            lat_spacing_deg=LATITUDE_SPACING_DEG,
            lng_spacing_deg=LONGITUDE_SPACING_DEG))

    (THIS_CENTROID_LAT_DEG, THIS_CENTROID_LNG_DEG
    ) = geodetic_utils.get_latlng_centroid(
        latitudes_deg=THESE_VERTEX_LATITUDES_DEG,
        longitudes_deg=THESE_VERTEX_LONGITUDES_DEG)

    STORM_OBJECT_TABLE_LARGE_SCALE[tracking_utils.CENTROID_LAT_COLUMN].values[
        i] = THIS_CENTROID_LAT_DEG
    STORM_OBJECT_TABLE_LARGE_SCALE[tracking_utils.CENTROID_LNG_COLUMN].values[
        i] = THIS_CENTROID_LNG_DEG

    STORM_OBJECT_TABLE_LARGE_SCALE[
        tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN].values[i] = (
            polygons.vertex_arrays_to_polygon_object(
                THESE_VERTEX_LONGITUDES_DEG, THESE_VERTEX_LATITUDES_DEG))
    STORM_OBJECT_TABLE_LARGE_SCALE[
        tracking_utils.POLYGON_OBJECT_ROWCOL_COLUMN].values[i] = (
            polygons.vertex_arrays_to_polygon_object(
                THESE_VERTEX_COLUMNS, THESE_VERTEX_ROWS))

STORM_OBJECT_TABLE_MERGED = copy.deepcopy(STORM_OBJECT_TABLE_SMALL_SCALE)
for this_column in tracking_utils.COLUMNS_TO_CHANGE_WHEN_MERGING_SCALES:
    STORM_OBJECT_TABLE_MERGED[this_column].values[
        2] = STORM_OBJECT_TABLE_LARGE_SCALE[this_column].values[1]


class StormTrackingUtilsTests(unittest.TestCase):
    """Each method is a unit test for storm_tracking_utils.py."""

    def test_check_data_source_segmotion(self):
        """Ensures correct output from check_data_source.

        In this case, data source is segmotion.
        """

        tracking_utils.check_data_source(tracking_utils.SEGMOTION_SOURCE_ID)

    def test_check_data_source_probsevere(self):
        """Ensures correct output from check_data_source.

        In this case, data source is probSevere.
        """

        tracking_utils.check_data_source(tracking_utils.PROBSEVERE_SOURCE_ID)

    def test_check_data_source_fake(self):
        """Ensures correct output from check_data_source.

        In this case, data source is unrecognized.
        """

        with self.assertRaises(ValueError):
            tracking_utils.check_data_source(FAKE_DATA_SOURCE)

    def test_remove_nan_rows_from_dataframe(self):
        """Ensures correct output from remove_nan_rows_from_dataframe."""

        this_table = tracking_utils.remove_nan_rows_from_dataframe(
            TABLE_WITH_NAN_ROWS)
        self.assertTrue(this_table.equals(TABLE_WITHOUT_NAN_ROWS))

    def test_column_name_to_distance_buffer_inclusive(self):
        """Ensures correct output from column_name_to_distance_buffer.

        In this case the distance buffer includes the storm object.
        """

        this_min_distance_metres, this_max_distance_metres = (
            tracking_utils.column_name_to_distance_buffer(
                BUFFER_COLUMN_NAME_INCLUSIVE))
        self.assertTrue(numpy.isnan(this_min_distance_metres))
        self.assertTrue(this_max_distance_metres == MAX_BUFFER_DISTANCE_METRES)

    def test_column_name_to_distance_buffer_exclusive(self):
        """Ensures correct output from column_name_to_distance_buffer.

        In this case the distance buffer does not include the storm object.
        """

        this_min_distance_metres, this_max_distance_metres = (
            tracking_utils.column_name_to_distance_buffer(
                BUFFER_COLUMN_NAME_EXCLUSIVE))
        self.assertTrue(this_min_distance_metres == MIN_BUFFER_DISTANCE_METRES)
        self.assertTrue(this_max_distance_metres == MAX_BUFFER_DISTANCE_METRES)

    def test_column_name_to_distance_buffer_fake(self):
        """Ensures correct output from column_name_to_distance_buffer.

        In this case the column name does not correspond to a distance buffer.
        """

        this_min_distance_metres, this_max_distance_metres = (
            tracking_utils.column_name_to_distance_buffer(
                BUFFER_COLUMN_NAME_FAKE))
        self.assertTrue(this_min_distance_metres is None)
        self.assertTrue(this_max_distance_metres is None)

    def test_distance_buffer_to_column_name_inclusive(self):
        """Ensures correct output from distance_buffer_to_column_name.

        In this case the distance buffer includes the storm object.
        """

        this_column_name = tracking_utils.distance_buffer_to_column_name(
            numpy.nan, MAX_BUFFER_DISTANCE_METRES)
        self.assertTrue(this_column_name == BUFFER_COLUMN_NAME_INCLUSIVE)

    def test_distance_buffer_to_column_name_exclusive(self):
        """Ensures correct output from distance_buffer_to_column_name.

        In this case the distance buffer does not include the storm object.
        """

        this_column_name = tracking_utils.distance_buffer_to_column_name(
            MIN_BUFFER_DISTANCE_METRES, MAX_BUFFER_DISTANCE_METRES)
        self.assertTrue(this_column_name == BUFFER_COLUMN_NAME_EXCLUSIVE)

    def test_get_grid_points_in_storms(self):
        """Ensures correct output from _get_grid_points_in_storms."""

        this_grid_points_in_storms_table = (
            tracking_utils._get_grid_points_in_storms(
                STORM_OBJECT_TABLE_SMALL_SCALE, num_grid_rows=NUM_GRID_ROWS,
                num_grid_columns=NUM_GRID_COLUMNS))

        self.assertTrue(this_grid_points_in_storms_table.equals(
            GRID_POINTS_IN_STORMS_TABLE))

    def test_merge_storms_at_two_scales(self):
        """Ensures correct output from merge_storms_at_two_scales."""

        this_storm_object_table_merged = (
            tracking_utils.merge_storms_at_two_scales(
                storm_object_table_small_scale=STORM_OBJECT_TABLE_SMALL_SCALE,
                storm_object_table_large_scale=STORM_OBJECT_TABLE_LARGE_SCALE,
                num_grid_rows=NUM_GRID_ROWS, num_grid_columns=NUM_GRID_COLUMNS))

        self.assertTrue(this_storm_object_table_merged.equals(
            STORM_OBJECT_TABLE_MERGED))


if __name__ == '__main__':
    unittest.main()
