"""Unit tests for storm_images.py"""

import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.deep_learning import storm_images

TOLERANCE = 1e-6

# The following constants are used to test _find_input_heights_needed.
STORM_ELEVATIONS_M_ASL = numpy.array(
    [309, 3691, 4269, 4257, 883, 685, 4800], dtype=float)
DESIRED_RADAR_HEIGHTS_M_AGL = numpy.array(
    [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000],
    dtype=int)

DESIRED_MYRORSS_HEIGHTS_M_ASL = numpy.array(
    [1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3500, 4000, 4500, 5000,
     5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 10000, 11000, 12000, 13000,
     14000, 15000, 16000, 17000],
    dtype=int)
DESIRED_GRIDRAD_HEIGHTS_M_ASL = numpy.linspace(1000, 14500, num=28, dtype=int)

# The following constants are used to test _fields_and_heights_to_pairs.
RADAR_FIELD_NAMES = [
    radar_utils.ECHO_TOP_40DBZ_NAME, radar_utils.LOW_LEVEL_SHEAR_NAME,
    radar_utils.REFL_NAME, radar_utils.VIL_NAME
]
REFLECTIVITY_HEIGHTS_M_AGL = numpy.array([1000, 2000], dtype=int)

FIELD_NAME_BY_PAIR_MYRORSS = [
    radar_utils.ECHO_TOP_40DBZ_NAME, radar_utils.LOW_LEVEL_SHEAR_NAME,
    radar_utils.REFL_NAME, radar_utils.REFL_NAME, radar_utils.VIL_NAME
]
HEIGHT_BY_PAIR_MYRORSS_M_AGL = numpy.array(
    [250, 250, 1000, 2000, 250], dtype=int)

FIELD_NAME_BY_PAIR_MRMS = FIELD_NAME_BY_PAIR_MYRORSS + []
HEIGHT_BY_PAIR_MRMS_M_AGL = numpy.array(
    [500, 250, 1000, 2000, 500], dtype=int)

# The following constants are used to test _get_relevant_storm_objects.
THESE_TIMES_UNIX_SEC = numpy.array([0, 0, 0, 1, 1, 1], dtype=int)
THESE_SPC_DATES_UNIX_SEC = numpy.array([2, 2, 3, 3, 4, 4], dtype=int)
THESE_EAST_VELOCITIES_M_S01 = numpy.array(
    [10, numpy.nan, 10, numpy.nan, 10, numpy.nan])
THESE_NORTH_VELOCITIES_M_S01 = numpy.array(
    [10, numpy.nan, 10, numpy.nan, 10, 10])

THIS_DICT = {
    tracking_utils.TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    tracking_utils.SPC_DATE_COLUMN: THESE_SPC_DATES_UNIX_SEC,
    tracking_utils.EAST_VELOCITY_COLUMN: THESE_EAST_VELOCITIES_M_S01,
    tracking_utils.NORTH_VELOCITY_COLUMN: THESE_NORTH_VELOCITIES_M_S01
}
STORM_OBJECT_TABLE = pandas.DataFrame.from_dict(THIS_DICT)

TIME_UNROTATED_GRID_UNIX_SEC = 0
SPC_DATE_UNROTATED_GRID_UNIX_SEC = 2
RELEVANT_INDICES_UNROTATED_GRID = numpy.array([0, 1], dtype=int)

FIRST_TIME_ROTATED_GRID_UNIX_SEC = 1
FIRST_DATE_ROTATED_GRID_UNIX_SEC = 3
FIRST_RELEVANT_INDICES_ROTATED_GRID = numpy.array([], dtype=int)

SECOND_TIME_ROTATED_GRID_UNIX_SEC = 1
SECOND_DATE_ROTATED_GRID_UNIX_SEC = 4
SECOND_RELEVANT_INDICES_ROTATED_GRID = numpy.array([4], dtype=int)

# The following constants are used to test _rotate_grid_one_storm_object.
ONE_CENTROID_LATITUDE_DEG = 53.5
ONE_CENTROID_LONGITUDE_DEG = 246.5
EASTWARD_MOTION_M_S01 = 1.
NORTHWARD_MOTION_M_S01 = 2.
NUM_FULL_GRID_ROWS_ROTATED = 4
NUM_FULL_GRID_COLUMNS_ROTATED = 6
ROTATED_GRID_SPACING_METRES = 1500.

ROTATED_LAT_MATRIX_ZERO_MOTION_DEG = numpy.array(
    [[53.480, 53.480, 53.480, 53.480, 53.480, 53.480],
     [53.493, 53.493, 53.493, 53.493, 53.493, 53.493],
     [53.507, 53.507, 53.507, 53.507, 53.507, 53.507],
     [53.520, 53.520, 53.520, 53.520, 53.520, 53.520]])
ROTATED_LNG_MATRIX_ZERO_MOTION_DEG = numpy.array(
    [[246.444, 246.466, 246.489, 246.511, 246.534, 246.556],
     [246.443, 246.466, 246.489, 246.511, 246.534, 246.557],
     [246.443, 246.466, 246.489, 246.511, 246.534, 246.557],
     [246.443, 246.466, 246.489, 246.511, 246.534, 246.557]])

ROTATED_LAT_MATRIX_EASTWARD_MOTION_DEG = ROTATED_LAT_MATRIX_ZERO_MOTION_DEG + 0.
ROTATED_LNG_MATRIX_EASTWARD_MOTION_DEG = ROTATED_LNG_MATRIX_ZERO_MOTION_DEG + 0.
ROTATED_LAT_MATRIX_WESTWARD_MOTION_DEG = numpy.flipud(
    ROTATED_LAT_MATRIX_ZERO_MOTION_DEG)
ROTATED_LNG_MATRIX_WESTWARD_MOTION_DEG = numpy.fliplr(
    ROTATED_LNG_MATRIX_ZERO_MOTION_DEG)

ROTATED_LAT_MATRIX_NORTHWARD_MOTION_DEG = numpy.array(
    [[53.466, 53.480, 53.493, 53.507, 53.520, 53.534],
     [53.466, 53.480, 53.493, 53.507, 53.520, 53.534],
     [53.466, 53.480, 53.493, 53.507, 53.520, 53.534],
     [53.466, 53.480, 53.493, 53.507, 53.520, 53.534]])
ROTATED_LNG_MATRIX_NORTHWARD_MOTION_DEG = numpy.array(
    [[246.534, 246.534, 246.534, 246.534, 246.534, 246.534],
     [246.511, 246.511, 246.511, 246.511, 246.511, 246.511],
     [246.489, 246.489, 246.489, 246.489, 246.489, 246.489],
     [246.466, 246.466, 246.466, 246.466, 246.466, 246.466]])

ROTATED_LAT_MATRIX_SOUTHWARD_MOTION_DEG = numpy.fliplr(
    ROTATED_LAT_MATRIX_NORTHWARD_MOTION_DEG)
ROTATED_LNG_MATRIX_SOUTHWARD_MOTION_DEG = numpy.flipud(
    ROTATED_LNG_MATRIX_NORTHWARD_MOTION_DEG)

ROTATED_LAT_MATRIX_ARBITRARY_MOTION_DEG = numpy.array(
    [[53.461, 53.473, 53.485, 53.497, 53.509, 53.521],
     [53.467, 53.479, 53.491, 53.503, 53.515, 53.527],
     [53.473, 53.485, 53.497, 53.509, 53.521, 53.533],
     [53.479, 53.491, 53.503, 53.515, 53.527, 53.539]])
ROTATED_LNG_MATRIX_ARBITRARY_MOTION_DEG = numpy.array(
    [[246.505, 246.515, 246.525, 246.535, 246.545, 246.556],
     [246.485, 246.495, 246.505, 246.515, 246.525, 246.535],
     [246.465, 246.475, 246.485, 246.495, 246.505, 246.515],
     [246.444, 246.455, 246.465, 246.475, 246.485, 246.495]])

# The following constants are used to test _centroids_latlng_to_rowcol.
NW_GRID_POINT_LAT_DEG = 55.
NW_GRID_POINT_LNG_DEG = 230.
LAT_SPACING_DEG = 0.01
LNG_SPACING_DEG = 0.01
NUM_FULL_GRID_ROWS = 3501
NUM_FULL_GRID_COLUMNS = 7001

CENTROID_LATITUDES_DEG = numpy.array([55., 54.995, 54.99, 54.985, 54.98])
CENTROID_LONGITUDES_DEG = numpy.array([230., 230.005, 230.01, 230.015, 230.02])
CENTER_ROWS = numpy.array([-0.5, 0.5, 1.5, 1.5, 1.5])
CENTER_COLUMNS = numpy.array([-0.5, 0.5, 1.5, 1.5, 1.5])

# The following constants are used to test _get_unrotated_storm_image_coords.
CENTER_ROW_TOP_LEFT = 10.5
CENTER_ROW_MIDDLE = 1000.5
CENTER_ROW_BOTTOM_RIGHT = 3490.5
CENTER_COLUMN_TOP_LEFT = 10.5
CENTER_COLUMN_MIDDLE = 1000.5
CENTER_COLUMN_BOTTOM_RIGHT = 6990.5
NUM_FULL_GRID_ROWS_UNROTATED = 32
NUM_FULL_GRID_COLUMNS_UNROTATED = 64

STORM_IMAGE_COORD_DICT_TOP_LEFT = {
    storm_images.FIRST_STORM_ROW_KEY: 0,
    storm_images.LAST_STORM_ROW_KEY: 26,
    storm_images.FIRST_STORM_COLUMN_KEY: 0,
    storm_images.LAST_STORM_COLUMN_KEY: 42,
    storm_images.NUM_TOP_PADDING_ROWS_KEY: 5,
    storm_images.NUM_BOTTOM_PADDING_ROWS_KEY: 0,
    storm_images.NUM_LEFT_PADDING_COLS_KEY: 21,
    storm_images.NUM_RIGHT_PADDING_COLS_KEY: 0
}

STORM_IMAGE_COORD_DICT_MIDDLE = {
    storm_images.FIRST_STORM_ROW_KEY: 985,
    storm_images.LAST_STORM_ROW_KEY: 1016,
    storm_images.FIRST_STORM_COLUMN_KEY: 969,
    storm_images.LAST_STORM_COLUMN_KEY: 1032,
    storm_images.NUM_TOP_PADDING_ROWS_KEY: 0,
    storm_images.NUM_BOTTOM_PADDING_ROWS_KEY: 0,
    storm_images.NUM_LEFT_PADDING_COLS_KEY: 0,
    storm_images.NUM_RIGHT_PADDING_COLS_KEY: 0
}

STORM_IMAGE_COORD_DICT_BOTTOM_RIGHT = {
    storm_images.FIRST_STORM_ROW_KEY: 3475,
    storm_images.LAST_STORM_ROW_KEY: 3500,
    storm_images.FIRST_STORM_COLUMN_KEY: 6959,
    storm_images.LAST_STORM_COLUMN_KEY: 7000,
    storm_images.NUM_TOP_PADDING_ROWS_KEY: 0,
    storm_images.NUM_BOTTOM_PADDING_ROWS_KEY: 6,
    storm_images.NUM_LEFT_PADDING_COLS_KEY: 0,
    storm_images.NUM_RIGHT_PADDING_COLS_KEY: 22
}

# The following constants are used to test _filter_storm_objects_by_label.
TORNADO_LABELS_TO_FILTER = numpy.array(
    [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0], dtype=int)

NUM_OBJECTS_BY_TORNADO_CLASS_DICT_NONZERO = {0: 2, 1: 50}
NUM_OBJECTS_BY_TORNADO_CLASS_DICT_ONE_ZERO = {0: 0, 1: 50}
INDICES_TO_KEEP_FOR_TORNADO_NONZERO = numpy.array([0, 1, 2, 4, 6], dtype=int)
INDICES_TO_KEEP_FOR_TORNADO_ONE_ZERO = numpy.array([2, 4, 6], dtype=int)

WIND_LABELS_TO_FILTER = numpy.array(
    [0, -2, 0, 5, 2, 1, 3, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 2, -2, 0, 1,
     3, 2, 4, 1, 0, 0, 1, 0, 0, -2, 4, -2, 0, 0, -2, 0], dtype=int)

NUM_OBJECTS_BY_WIND_CLASS_DICT_NONZERO = {
    -2: 5, 0: 1, 1: 2, 2: 3, 3: 4, 4: 25, 5: 100}
NUM_OBJECTS_BY_WIND_CLASS_DICT_SOME_ZERO = {
    -2: 0, 0: 1, 1: 0, 2: 3, 3: 0, 4: 25, 5: 0}

INDICES_TO_KEEP_FOR_WIND_NONZERO = numpy.array(
    [0, 5, 7, 4, 13, 20, 6, 18, 24, 26, 34, 3, 1, 21, 33, 35, 38], dtype=int)
INDICES_TO_KEEP_FOR_WIND_SOME_ZERO = numpy.array(
    [0, 4, 13, 20, 26, 34], dtype=int)

# The following constants are used to test _subset_xy_grid_for_interp.
NON_SUBSET_X_COORDS_METRES = numpy.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
NON_SUBSET_Y_COORDS_METRES = numpy.array([-10, -5, 0, 5, 10, 15], dtype=float)
NON_SUBSET_FIELD_MATRIX = numpy.array(
    [[1, 2, 3, 4, 5, 6, 7, 8],
     [2, 3, 4, 5, 6, 7, 8, 9],
     [3, 4, 5, 6, 7, 8, 9, 10],
     [4, 5, 6, 7, 8, 9, 10, 11],
     [5, 6, 7, 8, 9, 10, 11, 12],
     [6, 7, 8, 9, 10, 11, 12, 13]], dtype=float)

QUERY_X_COORDS_METRES = numpy.array(
    [3.5, 4.4, 2.1, 3.3, 3.8, 4.7, 4.4, 4.7, 3.0, 4.1])
QUERY_Y_COORDS_METRES = numpy.array(
    [4.1, -1.1, -0.7, -0.8, -2.2, 2.6, 1.4, -3.5, -1.0, 1.0])

SUBSET_X_COORDS_METRES = numpy.array([1, 2, 3, 4, 5, 6], dtype=float)
SUBSET_Y_COORDS_METRES = numpy.array([-10, -5, 0, 5, 10], dtype=float)
SUBSET_FIELD_MATRIX = numpy.array([[2, 3, 4, 5, 6, 7],
                                   [3, 4, 5, 6, 7, 8],
                                   [4, 5, 6, 7, 8, 9],
                                   [5, 6, 7, 8, 9, 10],
                                   [6, 7, 8, 9, 10, 11]], dtype=float)

# The following constants are used to test _extract_rotated_storm_image.
FULL_RADAR_MATRIX_ROTATED = numpy.array([[5, 5, 5, 5, 5, 5],
                                         [5, 5, 5, 5, 5, 5],
                                         [5, 5, 5, 5, 5, 5],
                                         [5, 5, 5, 5, 5, 5]], dtype=float)

NUM_SUBGRID_ROWS_ROTATED = 4
NUM_SUBGRID_COLUMNS_ROTATED = 6
FULL_GRID_POINT_LATITUDES_DEG = numpy.array([53.47, 53.49, 53.51, 53.53])
FULL_GRID_POINT_LONGITUDES_DEG = numpy.array(
    [246.45, 246.47, 246.49, 246.51, 246.53, 246.55])

STORM_IMAGE_MATRIX_ROTATED = numpy.array([[0, 5, 5, 5, 5, 0],
                                          [5, 5, 5, 5, 5, 0],
                                          [0, 5, 5, 5, 5, 5],
                                          [0, 5, 5, 5, 5, 0]], dtype=float)

# The following constants are used to test _extract_unrotated_storm_image.
FULL_RADAR_MATRIX_UNROTATED = numpy.array(
    [[numpy.nan, numpy.nan, 10, 20, 30, 40],
     [numpy.nan, 5, 15, 25, 35, 50],
     [5, 10, 25, 40, 55, 70],
     [10, 30, 50, 70, 75, 0]])

CENTER_ROW_TO_EXTRACT_MIDDLE = 1.5
CENTER_COLUMN_TO_EXTRACT_MIDDLE = 2.5
CENTER_ROW_TO_EXTRACT_EDGE = 3.5
CENTER_COLUMN_TO_EXTRACT_EDGE = 5.5
NUM_SUBGRID_ROWS_UNROTATED = 2
NUM_SUBGRID_COLUMNS_UNROTATED = 4

STORM_IMAGE_MATRIX_UNROTATED_MIDDLE = numpy.array(
    [[5, 15, 25, 35],
     [10, 25, 40, 55]], dtype=float)
STORM_IMAGE_MATRIX_UNROTATED_EDGE = numpy.array(
    [[75, 0, 0, 0],
     [0, 0, 0, 0]], dtype=float)

# The following constants are used to test _downsize_storm_images.
THIS_FIRST_MATRIX = numpy.array([[0, 1, 2, 3, 4, 5],
                                 [2, 3, 4, 5, 6, 7],
                                 [3, 4, 5, 6, 7, 8],
                                 [4, 5, 6, 7, 8, 9]], dtype=float)
THIS_SECOND_MATRIX = numpy.array([[0, 1, 2, 3, 4, 5],
                                  [6, 7, 8, 9, 10, 11],
                                  [12, 13, 14, 15, 16, 17],
                                  [18, 19, 20, 21, 22, 23]], dtype=float)
FULL_STORM_IMAGE_MATRIX = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0)

NUM_ROWS_TO_KEEP = 2
NUM_COLUMNS_TO_KEEP = 4

THIS_FIRST_MATRIX = numpy.array([[3, 4, 5, 6],
                                 [4, 5, 6, 7]], dtype=float)
THIS_SECOND_MATRIX = numpy.array([[7, 8, 9, 10],
                                  [13, 14, 15, 16]], dtype=float)
DOWNSIZED_STORM_IMAGE_MATRIX = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0)

# The following constants are used to test _interp_storm_image_in_height.
THIS_MATRIX_HEIGHT1 = numpy.array([[0, 1, 2, 3],
                                   [4, 5, 6, 7],
                                   [8, 9, 10, 11]], dtype=float)

ORIG_HEIGHTS_M_ASL = numpy.array([1000, 2000, 3000, 5000, 8000], dtype=int)
ORIG_IMAGE_MATRIX_3D = numpy.stack(
    (THIS_MATRIX_HEIGHT1, THIS_MATRIX_HEIGHT1 + 12, THIS_MATRIX_HEIGHT1 + 24,
     THIS_MATRIX_HEIGHT1 + 36, THIS_MATRIX_HEIGHT1 + 48),
    axis=-1)

INTERP_HEIGHTS_M_ASL = numpy.array(
    [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], dtype=int)
INTERP_IMAGE_MATRIX_3D = numpy.stack(
    (THIS_MATRIX_HEIGHT1, THIS_MATRIX_HEIGHT1 + 12, THIS_MATRIX_HEIGHT1 + 24,
     THIS_MATRIX_HEIGHT1 + 30, THIS_MATRIX_HEIGHT1 + 36,
     THIS_MATRIX_HEIGHT1 + 40, THIS_MATRIX_HEIGHT1 + 44,
     THIS_MATRIX_HEIGHT1 + 48, THIS_MATRIX_HEIGHT1 + 52,
     THIS_MATRIX_HEIGHT1 + 56),
    axis=-1)

# The following constants are used to test find_storm_image_file,
# find_storm_label_file, image_file_name_to_time, image_file_name_to_field,
# and image_file_name_to_height.
TOP_STORM_IMAGE_DIR_NAME = 'storm_images'
VALID_TIME_UNIX_SEC = 1516749825
SPC_DATE_STRING = '20180123'
RADAR_SOURCE_NAME = radar_utils.MYRORSS_SOURCE_ID
RADAR_FIELD_NAME = 'echo_top_40dbz_km'
RADAR_HEIGHT_M_AGL = 250

STORM_IMAGE_FILE_NAME_ONE_TIME = (
    'storm_images/myrorss/2018/20180123/echo_top_40dbz_km/00250_metres_agl/'
    'storm_images_2018-01-23-232345.nc')
STORM_IMAGE_FILE_NAME_ONE_SPC_DATE = (
    'storm_images/myrorss/2018/echo_top_40dbz_km/00250_metres_agl/'
    'storm_images_20180123.nc')

TOP_LABEL_DIR_NAME = 'labels'
LABEL_NAME = (
    'wind-speed_percentile=100.0_lead-time=0000-3600sec_distance=00000-10000m_'
    'cutoffs=10-20-30-40-50kt')

STORM_LABEL_FILE_NAME_ONE_TIME = (
    'labels/2018/20180123/wind_labels_2018-01-23-232345.nc')
STORM_LABEL_FILE_NAME_ONE_SPC_DATE = 'labels/2018/wind_labels_20180123.nc'


class StormImagesTests(unittest.TestCase):
    """Each method is a unit test for storm_images.py."""

    def test_find_input_heights_needed_myrorss(self):
        """Ensures correct output from _find_input_heights_needed.

        In this case the data source is MYRORSS.
        """

        these_heights_m_asl = storm_images._find_input_heights_needed(
            storm_elevations_m_asl=STORM_ELEVATIONS_M_ASL,
            desired_radar_heights_m_agl=DESIRED_RADAR_HEIGHTS_M_AGL,
            radar_source=radar_utils.MYRORSS_SOURCE_ID)

        self.assertTrue(numpy.array_equal(
            these_heights_m_asl, DESIRED_MYRORSS_HEIGHTS_M_ASL))

    def test_find_input_heights_needed_gridrad(self):
        """Ensures correct output from _find_input_heights_needed.

        In this case the data source is GridRad.
        """

        these_heights_m_asl = storm_images._find_input_heights_needed(
            storm_elevations_m_asl=STORM_ELEVATIONS_M_ASL,
            desired_radar_heights_m_agl=DESIRED_RADAR_HEIGHTS_M_AGL,
            radar_source=radar_utils.GRIDRAD_SOURCE_ID)

        self.assertTrue(numpy.array_equal(
            these_heights_m_asl, DESIRED_GRIDRAD_HEIGHTS_M_ASL))

    def test_fields_and_heights_to_pairs_myrorss(self):
        """Ensures correct output from _fields_and_heights_to_pairs.

        In this case the data source is MYRORSS.
        """

        (this_field_name_by_pair, this_height_by_pair_m_agl
        ) = storm_images._fields_and_heights_to_pairs(
            radar_field_names=RADAR_FIELD_NAMES,
            reflectivity_heights_m_agl=REFLECTIVITY_HEIGHTS_M_AGL,
            radar_source=radar_utils.MYRORSS_SOURCE_ID)

        self.assertTrue(this_field_name_by_pair == FIELD_NAME_BY_PAIR_MYRORSS)
        self.assertTrue(numpy.array_equal(
            this_height_by_pair_m_agl, HEIGHT_BY_PAIR_MYRORSS_M_AGL))

    def test_fields_and_heights_to_pairs_mrms(self):
        """Ensures correct output from _fields_and_heights_to_pairs.

        In this case the data source is MYRORSS.
        """

        (this_field_name_by_pair, this_height_by_pair_m_agl
        ) = storm_images._fields_and_heights_to_pairs(
            radar_field_names=RADAR_FIELD_NAMES,
            reflectivity_heights_m_agl=REFLECTIVITY_HEIGHTS_M_AGL,
            radar_source=radar_utils.MRMS_SOURCE_ID)

        self.assertTrue(this_field_name_by_pair == FIELD_NAME_BY_PAIR_MRMS)
        self.assertTrue(numpy.array_equal(
            this_height_by_pair_m_agl, HEIGHT_BY_PAIR_MRMS_M_AGL))

    def test_get_relevant_storm_objects_unrotated(self):
        """Ensures correct output from _get_relevant_storm_objects.

        In this case, storm-centered grids are unrotated.
        """

        these_indices = storm_images._get_relevant_storm_objects(
            storm_object_table=STORM_OBJECT_TABLE,
            valid_time_unix_sec=TIME_UNROTATED_GRID_UNIX_SEC,
            valid_spc_date_unix_sec=SPC_DATE_UNROTATED_GRID_UNIX_SEC,
            rotate_grids=False)

        self.assertTrue(numpy.array_equal(
            these_indices, RELEVANT_INDICES_UNROTATED_GRID))

    def test_get_relevant_storm_objects_rotated_first(self):
        """Ensures correct output from _get_relevant_storm_objects.

        In this case, storm-centered grids are rotated.
        """

        these_indices = storm_images._get_relevant_storm_objects(
            storm_object_table=STORM_OBJECT_TABLE,
            valid_time_unix_sec=FIRST_TIME_ROTATED_GRID_UNIX_SEC,
            valid_spc_date_unix_sec=FIRST_DATE_ROTATED_GRID_UNIX_SEC,
            rotate_grids=True)

        self.assertTrue(numpy.array_equal(
            these_indices, FIRST_RELEVANT_INDICES_ROTATED_GRID))

    def test_get_relevant_storm_objects_rotated_second(self):
        """Ensures correct output from _get_relevant_storm_objects.

        In this case, storm-centered grids are rotated.
        """

        these_indices = storm_images._get_relevant_storm_objects(
            storm_object_table=STORM_OBJECT_TABLE,
            valid_time_unix_sec=SECOND_TIME_ROTATED_GRID_UNIX_SEC,
            valid_spc_date_unix_sec=SECOND_DATE_ROTATED_GRID_UNIX_SEC,
            rotate_grids=True)

        self.assertTrue(numpy.array_equal(
            these_indices, SECOND_RELEVANT_INDICES_ROTATED_GRID))

    def test_rotate_grid_one_storm_object_zero_motion(self):
        """Ensures correct output from _rotate_grid_one_storm_object.

        In this case, storm motion is zero (stationary).
        """

        (this_latitude_matrix_deg, this_longitude_matrix_deg
        ) = storm_images._rotate_grid_one_storm_object(
            centroid_latitude_deg=ONE_CENTROID_LATITUDE_DEG,
            centroid_longitude_deg=ONE_CENTROID_LONGITUDE_DEG,
            eastward_motion_m_s01=0., northward_motion_m_s01=0.,
            num_storm_image_rows=NUM_FULL_GRID_ROWS_ROTATED,
            num_storm_image_columns=NUM_FULL_GRID_COLUMNS_ROTATED,
            storm_grid_spacing_metres=ROTATED_GRID_SPACING_METRES)

        self.assertTrue(numpy.allclose(
            this_latitude_matrix_deg, ROTATED_LAT_MATRIX_ZERO_MOTION_DEG,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            this_longitude_matrix_deg, ROTATED_LNG_MATRIX_ZERO_MOTION_DEG,
            atol=TOLERANCE))

    def test_rotate_grid_one_storm_object_eastward(self):
        """Ensures correct output from _rotate_grid_one_storm_object.

        In this case, storm motion is due eastward.
        """

        (this_latitude_matrix_deg, this_longitude_matrix_deg
        ) = storm_images._rotate_grid_one_storm_object(
            centroid_latitude_deg=ONE_CENTROID_LATITUDE_DEG,
            centroid_longitude_deg=ONE_CENTROID_LONGITUDE_DEG,
            eastward_motion_m_s01=EASTWARD_MOTION_M_S01,
            northward_motion_m_s01=0.,
            num_storm_image_rows=NUM_FULL_GRID_ROWS_ROTATED,
            num_storm_image_columns=NUM_FULL_GRID_COLUMNS_ROTATED,
            storm_grid_spacing_metres=ROTATED_GRID_SPACING_METRES)

        self.assertTrue(numpy.allclose(
            this_latitude_matrix_deg, ROTATED_LAT_MATRIX_EASTWARD_MOTION_DEG,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            this_longitude_matrix_deg, ROTATED_LNG_MATRIX_EASTWARD_MOTION_DEG,
            atol=TOLERANCE))

    def test_rotate_grid_one_storm_object_westward(self):
        """Ensures correct output from _rotate_grid_one_storm_object.

        In this case, storm motion is due westward.
        """

        (this_latitude_matrix_deg, this_longitude_matrix_deg
        ) = storm_images._rotate_grid_one_storm_object(
            centroid_latitude_deg=ONE_CENTROID_LATITUDE_DEG,
            centroid_longitude_deg=ONE_CENTROID_LONGITUDE_DEG,
            eastward_motion_m_s01=-EASTWARD_MOTION_M_S01,
            northward_motion_m_s01=0.,
            num_storm_image_rows=NUM_FULL_GRID_ROWS_ROTATED,
            num_storm_image_columns=NUM_FULL_GRID_COLUMNS_ROTATED,
            storm_grid_spacing_metres=ROTATED_GRID_SPACING_METRES)

        self.assertTrue(numpy.allclose(
            this_latitude_matrix_deg, ROTATED_LAT_MATRIX_WESTWARD_MOTION_DEG,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            this_longitude_matrix_deg, ROTATED_LNG_MATRIX_WESTWARD_MOTION_DEG,
            atol=TOLERANCE))

    def test_rotate_grid_one_storm_object_northward(self):
        """Ensures correct output from _rotate_grid_one_storm_object.

        In this case, storm motion is due northward.
        """

        (this_latitude_matrix_deg, this_longitude_matrix_deg
        ) = storm_images._rotate_grid_one_storm_object(
            centroid_latitude_deg=ONE_CENTROID_LATITUDE_DEG,
            centroid_longitude_deg=ONE_CENTROID_LONGITUDE_DEG,
            eastward_motion_m_s01=0.,
            northward_motion_m_s01=NORTHWARD_MOTION_M_S01,
            num_storm_image_rows=NUM_FULL_GRID_ROWS_ROTATED,
            num_storm_image_columns=NUM_FULL_GRID_COLUMNS_ROTATED,
            storm_grid_spacing_metres=ROTATED_GRID_SPACING_METRES)

        self.assertTrue(numpy.allclose(
            this_latitude_matrix_deg, ROTATED_LAT_MATRIX_NORTHWARD_MOTION_DEG,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            this_longitude_matrix_deg, ROTATED_LNG_MATRIX_NORTHWARD_MOTION_DEG,
            atol=TOLERANCE))

    def test_rotate_grid_one_storm_object_southward(self):
        """Ensures correct output from _rotate_grid_one_storm_object.

        In this case, storm motion is due southward.
        """

        (this_latitude_matrix_deg, this_longitude_matrix_deg
        ) = storm_images._rotate_grid_one_storm_object(
            centroid_latitude_deg=ONE_CENTROID_LATITUDE_DEG,
            centroid_longitude_deg=ONE_CENTROID_LONGITUDE_DEG,
            eastward_motion_m_s01=0.,
            northward_motion_m_s01=-NORTHWARD_MOTION_M_S01,
            num_storm_image_rows=NUM_FULL_GRID_ROWS_ROTATED,
            num_storm_image_columns=NUM_FULL_GRID_COLUMNS_ROTATED,
            storm_grid_spacing_metres=ROTATED_GRID_SPACING_METRES)

        self.assertTrue(numpy.allclose(
            this_latitude_matrix_deg, ROTATED_LAT_MATRIX_SOUTHWARD_MOTION_DEG,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            this_longitude_matrix_deg, ROTATED_LNG_MATRIX_SOUTHWARD_MOTION_DEG,
            atol=TOLERANCE))

    def test_rotate_grid_one_storm_object_arbitrary(self):
        """Ensures correct output from _rotate_grid_one_storm_object.

        In this case, storm motion is in an arbitrary direction (not 0, 90, 180,
        270, or NaN degrees).
        """

        (this_latitude_matrix_deg, this_longitude_matrix_deg
        ) = storm_images._rotate_grid_one_storm_object(
            centroid_latitude_deg=ONE_CENTROID_LATITUDE_DEG,
            centroid_longitude_deg=ONE_CENTROID_LONGITUDE_DEG,
            eastward_motion_m_s01=EASTWARD_MOTION_M_S01,
            northward_motion_m_s01=NORTHWARD_MOTION_M_S01,
            num_storm_image_rows=NUM_FULL_GRID_ROWS_ROTATED,
            num_storm_image_columns=NUM_FULL_GRID_COLUMNS_ROTATED,
            storm_grid_spacing_metres=ROTATED_GRID_SPACING_METRES)

        self.assertTrue(numpy.allclose(
            this_latitude_matrix_deg, ROTATED_LAT_MATRIX_ARBITRARY_MOTION_DEG,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            this_longitude_matrix_deg, ROTATED_LNG_MATRIX_ARBITRARY_MOTION_DEG,
            atol=TOLERANCE))

    def test_centroids_latlng_to_rowcol(self):
        """Ensures correct output from _centroids_latlng_to_rowcol."""

        (these_center_rows, these_center_columns
        ) = storm_images._centroids_latlng_to_rowcol(
            centroid_latitudes_deg=CENTROID_LATITUDES_DEG,
            centroid_longitudes_deg=CENTROID_LONGITUDES_DEG,
            nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
            nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
            lat_spacing_deg=LAT_SPACING_DEG,
            lng_spacing_deg=LNG_SPACING_DEG)

        self.assertTrue(numpy.array_equal(
            these_center_rows, CENTER_ROWS))
        self.assertTrue(numpy.array_equal(
            these_center_columns, CENTER_COLUMNS))

    def test_get_unrotated_storm_image_coords_top_left(self):
        """Ensures correct output from _get_unrotated_storm_image_coords.

        In this case, storm image runs off the full grid at the top-left.
        """

        this_coord_dict = storm_images._get_unrotated_storm_image_coords(
            num_full_grid_rows=NUM_FULL_GRID_ROWS,
            num_full_grid_columns=NUM_FULL_GRID_COLUMNS,
            num_storm_image_rows=NUM_FULL_GRID_ROWS_UNROTATED,
            num_storm_image_columns=NUM_FULL_GRID_COLUMNS_UNROTATED,
            center_row=CENTER_ROW_TOP_LEFT,
            center_column=CENTER_COLUMN_TOP_LEFT)

        self.assertTrue(this_coord_dict == STORM_IMAGE_COORD_DICT_TOP_LEFT)

    def test_get_unrotated_storm_image_coords_middle(self):
        """Ensures correct output from _get_unrotated_storm_image_coords.

        In this case, storm image does not run off the full grid.
        """

        this_coord_dict = storm_images._get_unrotated_storm_image_coords(
            num_full_grid_rows=NUM_FULL_GRID_ROWS,
            num_full_grid_columns=NUM_FULL_GRID_COLUMNS,
            num_storm_image_rows=NUM_FULL_GRID_ROWS_UNROTATED,
            num_storm_image_columns=NUM_FULL_GRID_COLUMNS_UNROTATED,
            center_row=CENTER_ROW_MIDDLE,
            center_column=CENTER_COLUMN_MIDDLE)

        self.assertTrue(this_coord_dict == STORM_IMAGE_COORD_DICT_MIDDLE)

    def test_get_unrotated_storm_image_coords_bottom_right(self):
        """Ensures correct output from _get_unrotated_storm_image_coords.

        In this case, storm image runs off the full grid at the bottom-right.
        """

        this_coord_dict = storm_images._get_unrotated_storm_image_coords(
            num_full_grid_rows=NUM_FULL_GRID_ROWS,
            num_full_grid_columns=NUM_FULL_GRID_COLUMNS,
            num_storm_image_rows=NUM_FULL_GRID_ROWS_UNROTATED,
            num_storm_image_columns=NUM_FULL_GRID_COLUMNS_UNROTATED,
            center_row=CENTER_ROW_BOTTOM_RIGHT,
            center_column=CENTER_COLUMN_BOTTOM_RIGHT)

        self.assertTrue(this_coord_dict == STORM_IMAGE_COORD_DICT_BOTTOM_RIGHT)

    def test_filter_storm_objects_by_label_tornado_nonzero(self):
        """Ensures correct output from _filter_storm_objects_by_label.

        In this case, target variable is tornado occurrence and desired number
        of storm objects is non-zero for all classes.
        """

        these_indices = storm_images._filter_storm_objects_by_label(
            label_values=TORNADO_LABELS_TO_FILTER,
            num_storm_objects_class_dict=
            NUM_OBJECTS_BY_TORNADO_CLASS_DICT_NONZERO, test_mode=True)

        self.assertTrue(numpy.array_equal(
            these_indices, INDICES_TO_KEEP_FOR_TORNADO_NONZERO))

    def test_filter_storm_objects_by_label_tornado_one_zero(self):
        """Ensures correct output from _filter_storm_objects_by_label.

        In this case, target variable is tornado occurrence and desired number
        of storm objects is zero for one class.
        """

        these_indices = storm_images._filter_storm_objects_by_label(
            label_values=TORNADO_LABELS_TO_FILTER,
            num_storm_objects_class_dict=
            NUM_OBJECTS_BY_TORNADO_CLASS_DICT_ONE_ZERO, test_mode=True)

        self.assertTrue(numpy.array_equal(
            these_indices, INDICES_TO_KEEP_FOR_TORNADO_ONE_ZERO))

    def test_filter_storm_objects_by_label_wind_nonzero(self):
        """Ensures correct output from _filter_storm_objects_by_label.

        In this case, target variable is wind-speed category and desired number
        of storm objects is non-zero for all classes.
        """

        these_indices = storm_images._filter_storm_objects_by_label(
            label_values=WIND_LABELS_TO_FILTER,
            num_storm_objects_class_dict=
            NUM_OBJECTS_BY_WIND_CLASS_DICT_NONZERO, test_mode=True)

        self.assertTrue(numpy.array_equal(
            these_indices, INDICES_TO_KEEP_FOR_WIND_NONZERO))

    def test_filter_storm_objects_by_label_wind_some_zero(self):
        """Ensures correct output from _filter_storm_objects_by_label.

        In this case, target variable is wind-speed category and desired number
        of storm objects is zero for some classes.
        """

        these_indices = storm_images._filter_storm_objects_by_label(
            label_values=WIND_LABELS_TO_FILTER,
            num_storm_objects_class_dict=
            NUM_OBJECTS_BY_WIND_CLASS_DICT_SOME_ZERO, test_mode=True)

        self.assertTrue(numpy.array_equal(
            these_indices, INDICES_TO_KEEP_FOR_WIND_SOME_ZERO))

    def test_subset_xy_grid_for_interp(self):
        """Ensures correct output from _subset_xy_grid_for_interp."""

        (this_field_matrix, these_x_coords_metres, these_y_coords_metres
        ) = storm_images._subset_xy_grid_for_interp(
            field_matrix=NON_SUBSET_FIELD_MATRIX,
            grid_point_x_coords_metres=NON_SUBSET_X_COORDS_METRES,
            grid_point_y_coords_metres=NON_SUBSET_Y_COORDS_METRES,
            query_x_coords_metres=QUERY_X_COORDS_METRES,
            query_y_coords_metres=QUERY_Y_COORDS_METRES)

        self.assertTrue(numpy.allclose(
            this_field_matrix, SUBSET_FIELD_MATRIX, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_x_coords_metres, SUBSET_X_COORDS_METRES, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_y_coords_metres, SUBSET_Y_COORDS_METRES, atol=TOLERANCE))

    def test_extract_rotated_storm_image(self):
        """Ensures correct output from _extract_rotated_storm_image."""

        this_storm_image_matrix = storm_images._extract_rotated_storm_image(
            full_radar_matrix=FULL_RADAR_MATRIX_ROTATED,
            full_grid_point_latitudes_deg=FULL_GRID_POINT_LATITUDES_DEG,
            full_grid_point_longitudes_deg=FULL_GRID_POINT_LONGITUDES_DEG,
            rotated_gp_lat_matrix_deg=ROTATED_LAT_MATRIX_ARBITRARY_MOTION_DEG,
            rotated_gp_lng_matrix_deg=ROTATED_LNG_MATRIX_ARBITRARY_MOTION_DEG)

        self.assertTrue(numpy.allclose(
            this_storm_image_matrix, STORM_IMAGE_MATRIX_ROTATED,
            atol=TOLERANCE))

    def test_extract_unrotated_storm_image_middle(self):
        """Ensures correct output from _extract_unrotated_storm_image.

        In this case the storm-centered image does *not* run off the full grid
        (no padding needed).
        """

        this_storm_image_matrix = storm_images._extract_unrotated_storm_image(
            full_radar_matrix=FULL_RADAR_MATRIX_UNROTATED,
            center_row=CENTER_ROW_TO_EXTRACT_MIDDLE,
            center_column=CENTER_COLUMN_TO_EXTRACT_MIDDLE,
            num_storm_image_rows=NUM_SUBGRID_ROWS_UNROTATED,
            num_storm_image_columns=NUM_SUBGRID_COLUMNS_UNROTATED)

        self.assertTrue(numpy.allclose(
            this_storm_image_matrix, STORM_IMAGE_MATRIX_UNROTATED_MIDDLE,
            atol=TOLERANCE))

    def test_extract_unrotated_storm_image_edge(self):
        """Ensures correct output from _extract_unrotated_storm_image.

        In this case the storm-centered image runs off the full grid (padding
        needed).
        """

        this_storm_image_matrix = storm_images._extract_unrotated_storm_image(
            full_radar_matrix=FULL_RADAR_MATRIX_UNROTATED,
            center_row=CENTER_ROW_TO_EXTRACT_EDGE,
            center_column=CENTER_COLUMN_TO_EXTRACT_EDGE,
            num_storm_image_rows=NUM_SUBGRID_ROWS_UNROTATED,
            num_storm_image_columns=NUM_SUBGRID_COLUMNS_UNROTATED)

        self.assertTrue(numpy.allclose(
            this_storm_image_matrix, STORM_IMAGE_MATRIX_UNROTATED_EDGE,
            atol=TOLERANCE))

    def test_downsize_storm_images_non_az_shear(self):
        """Ensures correct output from _downsize_storm_images.

        In this case, radar field is *not* azimuthal shear.
        """

        this_storm_image_matrix = storm_images._downsize_storm_images(
            storm_image_matrix=FULL_STORM_IMAGE_MATRIX,
            radar_field_name=radar_utils.REFL_NAME,
            num_rows_to_keep=NUM_ROWS_TO_KEEP,
            num_columns_to_keep=NUM_COLUMNS_TO_KEEP)

        self.assertTrue(numpy.allclose(
            this_storm_image_matrix, DOWNSIZED_STORM_IMAGE_MATRIX,
            atol=TOLERANCE))

    def test_downsize_storm_images_az_shear(self):
        """Ensures correct output from _downsize_storm_images.

        In this case, radar field is azimuthal shear.
        """

        this_storm_image_matrix = storm_images._downsize_storm_images(
            storm_image_matrix=FULL_STORM_IMAGE_MATRIX,
            radar_field_name=radar_utils.LOW_LEVEL_SHEAR_NAME,
            num_rows_to_keep=NUM_ROWS_TO_KEEP / 2,
            num_columns_to_keep=NUM_COLUMNS_TO_KEEP / 2)

        self.assertTrue(numpy.allclose(
            this_storm_image_matrix, DOWNSIZED_STORM_IMAGE_MATRIX,
            atol=TOLERANCE))

    def test_downsize_storm_images_no_storms(self):
        """Ensures correct output from _downsize_storm_images.

        In this case, there are no storm objects.
        """

        these_object_indices = numpy.array([], dtype=int)
        this_input_matrix = FULL_STORM_IMAGE_MATRIX[these_object_indices, ...]
        this_output_matrix = storm_images._downsize_storm_images(
            storm_image_matrix=this_input_matrix,
            radar_field_name=radar_utils.REFL_NAME,
            num_rows_to_keep=NUM_ROWS_TO_KEEP,
            num_columns_to_keep=NUM_COLUMNS_TO_KEEP)

        self.assertTrue(this_output_matrix.size == 0)

    def test_interp_storm_image_in_height(self):
        """Ensures correct output from _interp_storm_image_in_height."""

        this_interp_matrix = storm_images._interp_storm_image_in_height(
            storm_image_matrix_3d=ORIG_IMAGE_MATRIX_3D,
            orig_heights_m_asl=ORIG_HEIGHTS_M_ASL,
            new_heights_m_asl=INTERP_HEIGHTS_M_ASL)

        self.assertTrue(numpy.allclose(
            this_interp_matrix, INTERP_IMAGE_MATRIX_3D, atol=TOLERANCE))

    def test_find_storm_image_file_one_time(self):
        """Ensures correct output from find_storm_image_file.

        In this case, file name is for one time step.
        """

        this_file_name = storm_images.find_storm_image_file(
            top_directory_name=TOP_STORM_IMAGE_DIR_NAME,
            unix_time_sec=VALID_TIME_UNIX_SEC, spc_date_string=SPC_DATE_STRING,
            radar_source=RADAR_SOURCE_NAME, radar_field_name=RADAR_FIELD_NAME,
            radar_height_m_agl=RADAR_HEIGHT_M_AGL, raise_error_if_missing=False)

        self.assertTrue(this_file_name == STORM_IMAGE_FILE_NAME_ONE_TIME)

    def test_find_storm_image_file_one_spc_date(self):
        """Ensures correct output from find_storm_image_file.

        In this case, file name is for one SPC date.
        """

        this_file_name = storm_images.find_storm_image_file(
            top_directory_name=TOP_STORM_IMAGE_DIR_NAME,
            spc_date_string=SPC_DATE_STRING, radar_source=RADAR_SOURCE_NAME,
            radar_field_name=RADAR_FIELD_NAME,
            radar_height_m_agl=RADAR_HEIGHT_M_AGL, raise_error_if_missing=False)

        self.assertTrue(this_file_name == STORM_IMAGE_FILE_NAME_ONE_SPC_DATE)

    def test_image_file_name_to_time_one_time(self):
        """Ensures correct output from image_file_name_to_time.

        In this case, file name is for one time step.
        """

        (this_time_unix_sec, this_spc_date_string
        ) = storm_images.image_file_name_to_time(STORM_IMAGE_FILE_NAME_ONE_TIME)

        self.assertTrue(this_time_unix_sec == VALID_TIME_UNIX_SEC)
        self.assertTrue(this_spc_date_string == SPC_DATE_STRING)

    def test_image_file_name_to_time_one_spc_date(self):
        """Ensures correct output from image_file_name_to_time.

        In this case, file name is for one SPC date.
        """

        (this_time_unix_sec, this_spc_date_string
        ) = storm_images.image_file_name_to_time(
            STORM_IMAGE_FILE_NAME_ONE_SPC_DATE)

        self.assertTrue(this_time_unix_sec is None)
        self.assertTrue(this_spc_date_string == SPC_DATE_STRING)

    def test_image_file_name_to_field_one_time(self):
        """Ensures correct output from image_file_name_to_field.

        In this case, file name is for one time step.
        """

        this_field_name = storm_images.image_file_name_to_field(
            STORM_IMAGE_FILE_NAME_ONE_TIME)
        self.assertTrue(this_field_name == RADAR_FIELD_NAME)

    def test_image_file_name_to_field_one_spc_date(self):
        """Ensures correct output from image_file_name_to_field.

        In this case, file name is for one SPC date.
        """

        this_field_name = storm_images.image_file_name_to_field(
            STORM_IMAGE_FILE_NAME_ONE_SPC_DATE)
        self.assertTrue(this_field_name == RADAR_FIELD_NAME)

    def test_image_file_name_to_height_one_time(self):
        """Ensures correct output from image_file_name_to_height.

        In this case, file name is for one time step.
        """

        this_height_m_agl = storm_images.image_file_name_to_height(
            STORM_IMAGE_FILE_NAME_ONE_TIME)
        self.assertTrue(this_height_m_agl == RADAR_HEIGHT_M_AGL)

    def test_image_file_name_to_height_one_spc_date(self):
        """Ensures correct output from image_file_name_to_height.

        In this case, file name is for one SPC date.
        """

        this_height_m_agl = storm_images.image_file_name_to_height(
            STORM_IMAGE_FILE_NAME_ONE_SPC_DATE)
        self.assertTrue(this_height_m_agl == RADAR_HEIGHT_M_AGL)

    def test_find_storm_label_file_one_time(self):
        """Ensures correct output from find_storm_label_file.

        In this case, file name is for one time step.
        """

        this_file_name = storm_images.find_storm_label_file(
            storm_image_file_name=STORM_IMAGE_FILE_NAME_ONE_TIME,
            top_label_dir_name=TOP_LABEL_DIR_NAME, label_name=LABEL_NAME,
            raise_error_if_missing=False, warn_if_missing=False)

        self.assertTrue(this_file_name == STORM_LABEL_FILE_NAME_ONE_TIME)

    def test_find_storm_label_file_one_spc_date(self):
        """Ensures correct output from find_storm_label_file.

        In this case, file name is for one SPC date.
        """

        this_file_name = storm_images.find_storm_label_file(
            storm_image_file_name=STORM_IMAGE_FILE_NAME_ONE_SPC_DATE,
            top_label_dir_name=TOP_LABEL_DIR_NAME, label_name=LABEL_NAME,
            raise_error_if_missing=False, warn_if_missing=False)

        self.assertTrue(this_file_name == STORM_LABEL_FILE_NAME_ONE_SPC_DATE)


if __name__ == '__main__':
    unittest.main()
