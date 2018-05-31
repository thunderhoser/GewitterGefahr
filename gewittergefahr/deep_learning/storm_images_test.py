"""Unit tests for storm_images.py"""

import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import link_events_to_storms as events2storms
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.deep_learning import storm_images

TOLERANCE = 1e-6

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

# The following constants are used to test _get_storm_image_coords.
CENTER_ROW_TOP_LEFT = 10.5
CENTER_ROW_MIDDLE = 1000.5
CENTER_ROW_BOTTOM_RIGHT = 3490.5
CENTER_COLUMN_TOP_LEFT = 10.5
CENTER_COLUMN_MIDDLE = 1000.5
CENTER_COLUMN_BOTTOM_RIGHT = 6990.5
NUM_STORM_IMAGE_ROWS = 32
NUM_STORM_IMAGE_COLUMNS = 64

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

# The following constants are used to test _check_storm_labels.
MIN_LEAD_TIME_SEC = 0
MAX_LEAD_TIME_SEC = 900
MIN_LINK_DISTANCE_METRES = 1
MAX_LINK_DISTANCE_METRES = 5000
WIND_SPEED_PERCENTILE_LEVEL = 100.
WIND_SPEED_CUTOFFS_KT = numpy.array([50.])

WIND_SPEED_REGRESSION_COLUMN = labels.get_column_name_for_regression_label(
    min_lead_time_sec=MIN_LEAD_TIME_SEC, max_lead_time_sec=MAX_LEAD_TIME_SEC,
    min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
    max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
    wind_speed_percentile_level=WIND_SPEED_PERCENTILE_LEVEL)

NUM_WIND_OBS_COLUMN = labels.get_column_name_for_num_wind_obs(
    min_lead_time_sec=MIN_LEAD_TIME_SEC, max_lead_time_sec=MAX_LEAD_TIME_SEC,
    min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
    max_link_distance_metres=MAX_LINK_DISTANCE_METRES)

WIND_SPEED_LABEL_COLUMN = labels.get_column_name_for_classification_label(
    min_lead_time_sec=MIN_LEAD_TIME_SEC, max_lead_time_sec=MAX_LEAD_TIME_SEC,
    min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
    max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
    event_type_string=events2storms.WIND_EVENT_TYPE_STRING,
    wind_speed_percentile_level=WIND_SPEED_PERCENTILE_LEVEL,
    wind_speed_class_cutoffs_kt=WIND_SPEED_CUTOFFS_KT)

TORNADO_LABEL_COLUMN = labels.get_column_name_for_classification_label(
    min_lead_time_sec=MIN_LEAD_TIME_SEC, max_lead_time_sec=MAX_LEAD_TIME_SEC,
    min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
    max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
    event_type_string=events2storms.TORNADO_EVENT_TYPE_STRING)

STORM_IDS = ['A', 'B', 'C', 'A', 'B', 'C']
UNIX_TIMES_SEC = numpy.array([0, 0, 0, 1, 1, 1], dtype=int)
MAX_WIND_SPEEDS_KT = numpy.array([10., 25., 50., 7., 55., 40.])
WIND_OB_COUNTS = numpy.array([5, 50, 3, 3, 42, 6], dtype=int)
WIND_SPEED_LABELS = numpy.array([0, 0, 1, 0, 1, 0], dtype=int)
TORNADO_LABELS = numpy.array([1, 0, 0, 1, 1, 0], dtype=int)

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: STORM_IDS,
    tracking_utils.TIME_COLUMN: UNIX_TIMES_SEC,
    NUM_WIND_OBS_COLUMN: WIND_OB_COUNTS,
    WIND_SPEED_REGRESSION_COLUMN: MAX_WIND_SPEEDS_KT,
    WIND_SPEED_LABEL_COLUMN: WIND_SPEED_LABELS
}
STORM_TO_WINDS_TABLE = pandas.DataFrame.from_dict(THIS_DICT)
STORM_TO_WINDS_TABLE_AB_TIME0 = STORM_TO_WINDS_TABLE.iloc[[0, 1]]
STORM_TO_WINDS_TABLE_CB_TIME1 = STORM_TO_WINDS_TABLE.iloc[[5, 4]]

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: STORM_IDS,
    tracking_utils.TIME_COLUMN: UNIX_TIMES_SEC,
    TORNADO_LABEL_COLUMN: TORNADO_LABELS
}
STORM_TO_TORNADOES_TABLE = pandas.DataFrame.from_dict(THIS_DICT)
STORM_TO_TORNADOES_TABLE_AB_TIME0 = STORM_TO_TORNADOES_TABLE.iloc[[0, 1]]
STORM_TO_TORNADOES_TABLE_CB_TIME1 = STORM_TO_TORNADOES_TABLE.iloc[[5, 4]]

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

# The following constants are used to test _find_storm_objects.
ALL_STORM_IDS = ['a', 'b', 'c', 'd', 'a', 'c', 'e', 'f', 'e']
ALL_VALID_TIMES_UNIX_SEC = numpy.array([0, 0, 0, 0, 1, 1, 1, 1, 2], dtype=int)
STORM_IDS_TO_KEEP = ['a', 'c', 'a', 'e', 'e']
VALID_TIMES_TO_KEEP_UNIX_SEC = numpy.array([0, 0, 1, 1, 2], dtype=int)
RELEVANT_INDICES = numpy.array([0, 2, 4, 6, 8], dtype=int)

# The following constants are used to test extract_one_label_per_storm.
WIND_SPEED_LABELS_CB_TIME1 = numpy.array([0, 1], dtype=int)
TORNADO_LABELS_CB_TIME1 = numpy.array([0, 1], dtype=int)
WIND_SPEED_LABELS_AB_TIME0 = numpy.array([0, 0], dtype=int)
TORNADO_LABELS_AB_TIME0 = numpy.array([1, 0], dtype=int)

# The following constants are used to test extract_storm_image.
FULL_RADAR_MATRIX = numpy.array(
    [[numpy.nan, numpy.nan, 10., 20., 30., 40.],
     [numpy.nan, 5., 15., 25., 35., 50.],
     [5., 10., 25., 40., 55., 70.],
     [10., 30., 50., 70., 75., numpy.nan]])

CENTER_ROW_TO_EXTRACT_MIDDLE = 1.5
CENTER_COLUMN_TO_EXTRACT_MIDDLE = 2.5
CENTER_ROW_TO_EXTRACT_EDGE = 3.5
CENTER_COLUMN_TO_EXTRACT_EDGE = 5.5
NUM_ROWS_TO_EXTRACT = 2
NUM_COLUMNS_TO_EXTRACT = 4

STORM_IMAGE_MATRIX_MIDDLE = numpy.array([[5., 15., 25., 35.],
                                         [10., 25., 40., 55.]])
STORM_IMAGE_MATRIX_EDGE = numpy.array([[75., numpy.nan, 0., 0.],
                                       [0., 0., 0., 0.]])

# The following constants are used to test find_storm_image_file,
# image_file_name_to_time, and find_storm_label_file.
TOP_STORM_IMAGE_DIR_NAME = 'storm_images'
VALID_TIME_UNIX_SEC = 1516749825
SPC_DATE_STRING = '20180123'
RADAR_SOURCE_NAME = radar_utils.MYRORSS_SOURCE_ID
RADAR_FIELD_NAME = 'echo_top_40dbz_km'
RADAR_HEIGHT_M_ASL = 250

STORM_IMAGE_FILE_NAME_ONE_TIME = (
    'storm_images/myrorss/2018/20180123/echo_top_40dbz_km/00250_metres_asl/'
    'storm_images_2018-01-23-232345.nc')
STORM_IMAGE_FILE_NAME_ONE_SPC_DATE = (
    'storm_images/myrorss/2018/echo_top_40dbz_km/00250_metres_asl/'
    'storm_images_20180123.nc')

TOP_LABEL_DIRECTORY_NAME = 'labels'
LABEL_NAME = (
    'wind-speed_percentile=100.0_lead-time=0000-3600sec_distance=00000-10000m_'
    'cutoffs=10-20-30-40-50kt')

STORM_LABEL_FILE_NAME_ONE_TIME = (
    'labels/2018/20180123/wind_labels_2018-01-23-232345.p')
STORM_LABEL_FILE_NAME_ONE_SPC_DATE = 'labels/2018/wind_labels_20180123.p'


class StormImagesTests(unittest.TestCase):
    """Each method is a unit test for storm_images.py."""

    def test_centroids_latlng_to_rowcol(self):
        """Ensures correct output from _centroids_latlng_to_rowcol."""

        these_center_rows, these_center_columns = (
            storm_images._centroids_latlng_to_rowcol(
                centroid_latitudes_deg=CENTROID_LATITUDES_DEG,
                centroid_longitudes_deg=CENTROID_LONGITUDES_DEG,
                nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
                nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
                lat_spacing_deg=LAT_SPACING_DEG,
                lng_spacing_deg=LNG_SPACING_DEG))

        self.assertTrue(numpy.array_equal(
            these_center_rows, CENTER_ROWS))
        self.assertTrue(numpy.array_equal(
            these_center_columns, CENTER_COLUMNS))

    def test_get_storm_image_coords_top_left(self):
        """Ensures correct output from _get_storm_image_coords.

        In this case, storm image runs off the full grid at the top-left.
        """

        this_coord_dict = storm_images._get_storm_image_coords(
            num_full_grid_rows=NUM_FULL_GRID_ROWS,
            num_full_grid_columns=NUM_FULL_GRID_COLUMNS,
            num_storm_image_rows=NUM_STORM_IMAGE_ROWS,
            num_storm_image_columns=NUM_STORM_IMAGE_COLUMNS,
            center_row=CENTER_ROW_TOP_LEFT,
            center_column=CENTER_COLUMN_TOP_LEFT)

        self.assertTrue(this_coord_dict == STORM_IMAGE_COORD_DICT_TOP_LEFT)

    def test_get_storm_image_coords_middle(self):
        """Ensures correct output from _get_storm_image_coords.

        In this case, storm image does not run off the full grid.
        """

        this_coord_dict = storm_images._get_storm_image_coords(
            num_full_grid_rows=NUM_FULL_GRID_ROWS,
            num_full_grid_columns=NUM_FULL_GRID_COLUMNS,
            num_storm_image_rows=NUM_STORM_IMAGE_ROWS,
            num_storm_image_columns=NUM_STORM_IMAGE_COLUMNS,
            center_row=CENTER_ROW_MIDDLE,
            center_column=CENTER_COLUMN_MIDDLE)

        self.assertTrue(this_coord_dict == STORM_IMAGE_COORD_DICT_MIDDLE)

    def test_get_storm_image_coords_bottom_right(self):
        """Ensures correct output from _get_storm_image_coords.

        In this case, storm image runs off the full grid at the bottom-right.
        """

        this_coord_dict = storm_images._get_storm_image_coords(
            num_full_grid_rows=NUM_FULL_GRID_ROWS,
            num_full_grid_columns=NUM_FULL_GRID_COLUMNS,
            num_storm_image_rows=NUM_STORM_IMAGE_ROWS,
            num_storm_image_columns=NUM_STORM_IMAGE_COLUMNS,
            center_row=CENTER_ROW_BOTTOM_RIGHT,
            center_column=CENTER_COLUMN_BOTTOM_RIGHT)

        self.assertTrue(this_coord_dict == STORM_IMAGE_COORD_DICT_BOTTOM_RIGHT)

    def test_check_storm_labels_ab_time0(self):
        """Ensures correct output from _check_storm_labels.

        In this case, input storm objects are A and B at time 0.
        """

        this_storm_to_winds_table, storm_to_tornadoes_table = (
            storm_images._check_storm_labels(
                storm_ids=['A', 'B'],
                valid_times_unix_sec=numpy.full(2, 0, dtype=int),
                storm_to_winds_table=STORM_TO_WINDS_TABLE,
                storm_to_tornadoes_table=STORM_TO_TORNADOES_TABLE))

        self.assertTrue(this_storm_to_winds_table.equals(
            STORM_TO_WINDS_TABLE_AB_TIME0))
        self.assertTrue(storm_to_tornadoes_table.equals(
            STORM_TO_TORNADOES_TABLE_AB_TIME0))

    def test_check_storm_labels_bc_time1(self):
        """Ensures correct output from _check_storm_labels.

        In this case, input storm objects are B and C at time 1.
        """

        this_storm_to_winds_table, this_storm_to_tornadoes_table = (
            storm_images._check_storm_labels(
                storm_ids=['C', 'B'],
                valid_times_unix_sec=numpy.full(2, 1, dtype=int),
                storm_to_winds_table=STORM_TO_WINDS_TABLE,
                storm_to_tornadoes_table=STORM_TO_TORNADOES_TABLE))

        self.assertTrue(this_storm_to_winds_table.equals(
            STORM_TO_WINDS_TABLE_CB_TIME1))
        self.assertTrue(this_storm_to_tornadoes_table.equals(
            STORM_TO_TORNADOES_TABLE_CB_TIME1))

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

    def test_find_storm_objects(self):
        """Ensures correct output from _find_storm_objects."""

        these_indices = storm_images._find_storm_objects(
            all_storm_ids=ALL_STORM_IDS,
            all_valid_times_unix_sec=ALL_VALID_TIMES_UNIX_SEC,
            storm_ids_to_keep=STORM_IDS_TO_KEEP,
            valid_times_to_keep_unix_sec=VALID_TIMES_TO_KEEP_UNIX_SEC)
        self.assertTrue(numpy.array_equal(these_indices, RELEVANT_INDICES))

    def test_extract_storm_image_middle(self):
        """Ensures correct output from extract_storm_image.

        In this case, storm image does not run off the full grid (no padding
        needed).
        """

        this_storm_image_matrix = storm_images.extract_storm_image(
            full_radar_matrix=FULL_RADAR_MATRIX,
            center_row=CENTER_ROW_TO_EXTRACT_MIDDLE,
            center_column=CENTER_COLUMN_TO_EXTRACT_MIDDLE,
            num_storm_image_rows=NUM_ROWS_TO_EXTRACT,
            num_storm_image_columns=NUM_COLUMNS_TO_EXTRACT)

        self.assertTrue(numpy.allclose(
            this_storm_image_matrix, STORM_IMAGE_MATRIX_MIDDLE, atol=TOLERANCE))

    def test_extract_storm_image_edge(self):
        """Ensures correct output from extract_storm_image.

        In this case, storm image runs off the full grid (padding needed).
        """

        this_storm_image_matrix = storm_images.extract_storm_image(
            full_radar_matrix=FULL_RADAR_MATRIX,
            center_row=CENTER_ROW_TO_EXTRACT_EDGE,
            center_column=CENTER_COLUMN_TO_EXTRACT_EDGE,
            num_storm_image_rows=NUM_ROWS_TO_EXTRACT,
            num_storm_image_columns=NUM_COLUMNS_TO_EXTRACT)

        self.assertTrue(numpy.allclose(
            this_storm_image_matrix, STORM_IMAGE_MATRIX_EDGE, atol=TOLERANCE,
            equal_nan=True))

    def test_find_storm_image_file_one_time(self):
        """Ensures correct output from find_storm_image_file.

        In this case, file name is for one time step.
        """

        this_file_name = storm_images.find_storm_image_file(
            top_directory_name=TOP_STORM_IMAGE_DIR_NAME,
            unix_time_sec=VALID_TIME_UNIX_SEC, spc_date_string=SPC_DATE_STRING,
            radar_source=RADAR_SOURCE_NAME, radar_field_name=RADAR_FIELD_NAME,
            radar_height_m_asl=RADAR_HEIGHT_M_ASL, raise_error_if_missing=False)

        self.assertTrue(this_file_name == STORM_IMAGE_FILE_NAME_ONE_TIME)

    def test_find_storm_image_file_one_spc_date(self):
        """Ensures correct output from find_storm_image_file.

        In this case, file name is for one SPC date.
        """

        this_file_name = storm_images.find_storm_image_file(
            top_directory_name=TOP_STORM_IMAGE_DIR_NAME,
            spc_date_string=SPC_DATE_STRING, radar_source=RADAR_SOURCE_NAME,
            radar_field_name=RADAR_FIELD_NAME,
            radar_height_m_asl=RADAR_HEIGHT_M_ASL, raise_error_if_missing=False)

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

    def test_find_storm_label_file_one_time(self):
        """Ensures correct output from find_storm_label_file.

        In this case, file name is for one time step.
        """

        this_file_name = storm_images.find_storm_label_file(
            storm_image_file_name=STORM_IMAGE_FILE_NAME_ONE_TIME,
            top_label_directory_name=TOP_LABEL_DIRECTORY_NAME,
            label_name=LABEL_NAME, raise_error_if_missing=False)

        self.assertTrue(this_file_name == STORM_LABEL_FILE_NAME_ONE_TIME)

    def test_find_storm_label_file_one_spc_date(self):
        """Ensures correct output from find_storm_label_file.

        In this case, file name is for one SPC date.
        """

        this_file_name = storm_images.find_storm_label_file(
            storm_image_file_name=STORM_IMAGE_FILE_NAME_ONE_TIME,
            top_label_directory_name=TOP_LABEL_DIRECTORY_NAME,
            label_name=LABEL_NAME, raise_error_if_missing=False)

        self.assertTrue(this_file_name == STORM_LABEL_FILE_NAME_ONE_TIME)

    def test_extract_storm_labels_with_name_wind_ab_time0(self):
        """Ensures correct output from extract_storm_labels_with_name.

        In this case, extracting wind labels for storms A and B at time 0.
        """

        these_labels = storm_images.extract_storm_labels_with_name(
            storm_ids=['A', 'B'],
            valid_times_unix_sec=numpy.full(2, 0, dtype=int),
            label_name=WIND_SPEED_LABEL_COLUMN,
            storm_to_winds_table=STORM_TO_WINDS_TABLE_AB_TIME0)
        self.assertTrue(numpy.array_equal(
            these_labels, WIND_SPEED_LABELS_AB_TIME0))

    def test_extract_storm_labels_with_name_tornado_ab_time0(self):
        """Ensures correct output from extract_storm_labels_with_name.

        In this case, extracting tornado labels for storms A and B at time 0.
        """

        these_labels = storm_images.extract_storm_labels_with_name(
            storm_ids=['A', 'B'],
            valid_times_unix_sec=numpy.full(2, 0, dtype=int),
            label_name=TORNADO_LABEL_COLUMN,
            storm_to_tornadoes_table=STORM_TO_TORNADOES_TABLE_AB_TIME0)
        self.assertTrue(numpy.array_equal(
            these_labels, TORNADO_LABELS_AB_TIME0))

    def test_extract_storm_labels_with_name_wind_cb_time1(self):
        """Ensures correct output from extract_storm_labels_with_name.

        In this case, extracting wind labels for storms C and B at time 1.
        """

        these_labels = storm_images.extract_storm_labels_with_name(
            storm_ids=['C', 'B'],
            valid_times_unix_sec=numpy.full(2, 1, dtype=int),
            label_name=WIND_SPEED_LABEL_COLUMN,
            storm_to_winds_table=STORM_TO_WINDS_TABLE_CB_TIME1)
        self.assertTrue(numpy.array_equal(
            these_labels, WIND_SPEED_LABELS_CB_TIME1))

    def test_extract_storm_labels_with_name_tornado_cb_time1(self):
        """Ensures correct output from extract_storm_labels_with_name.

        In this case, extracting tornado labels for storms C and B at time 1.
        """

        these_labels = storm_images.extract_storm_labels_with_name(
            storm_ids=['C', 'B'],
            valid_times_unix_sec=numpy.full(2, 1, dtype=int),
            label_name=TORNADO_LABEL_COLUMN,
            storm_to_tornadoes_table=STORM_TO_TORNADOES_TABLE_CB_TIME1)
        self.assertTrue(numpy.array_equal(
            these_labels, TORNADO_LABELS_CB_TIME1))


if __name__ == '__main__':
    unittest.main()
