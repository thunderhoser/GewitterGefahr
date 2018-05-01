"""Methods for handling storm images.

A "storm image" is a radar image that shares a center with the storm object (in
other words, the center of the image is the centroid of the storm object).
"""

import os
import copy
import glob
import pickle
import numpy
from gewittergefahr.gg_io import gridrad_io
from gewittergefahr.gg_io import myrorss_and_mrms_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import gridrad_utils
from gewittergefahr.gg_utils import myrorss_and_mrms_utils
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import link_events_to_storms as events2storms
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

PADDING_VALUE = 0

DAYS_TO_SECONDS = 86400
GRIDRAD_TIME_INTERVAL_SEC = 300
TIME_FORMAT = '%Y-%m-%d-%H%M%S'
TIME_FORMAT_REGEX = (
    '[0-9][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9]')

FIRST_STORM_ROW_KEY = 'first_storm_image_row'
LAST_STORM_ROW_KEY = 'last_storm_image_row'
FIRST_STORM_COLUMN_KEY = 'first_storm_image_column'
LAST_STORM_COLUMN_KEY = 'last_storm_image_column'
NUM_TOP_PADDING_ROWS_KEY = 'num_padding_rows_at_top'
NUM_BOTTOM_PADDING_ROWS_KEY = 'num_padding_rows_at_bottom'
NUM_LEFT_PADDING_COLS_KEY = 'num_padding_columns_at_left'
NUM_RIGHT_PADDING_COLS_KEY = 'num_padding_columns_at_right'

STORM_IMAGE_MATRIX_KEY = 'storm_image_matrix'
STORM_IDS_KEY = 'storm_ids'
VALID_TIME_KEY = 'unix_time_sec'
RADAR_FIELD_NAME_KEY = 'radar_field_name'
RADAR_HEIGHT_KEY = 'radar_height_m_asl'
STORM_TO_WINDS_TABLE_KEY = 'storm_to_winds_table'
STORM_TO_TORNADOES_TABLE_KEY = 'storm_to_tornadoes_table'

DEFAULT_NUM_IMAGE_ROWS = 32
DEFAULT_NUM_IMAGE_COLUMNS = 32
# MIN_NUM_IMAGE_ROWS = 8
# MIN_NUM_IMAGE_COLUMNS = 8
MIN_NUM_IMAGE_ROWS = 2
MIN_NUM_IMAGE_COLUMNS = 2

DEFAULT_MYRORSS_MRMS_FIELD_NAMES = [
    radar_utils.ECHO_TOP_18DBZ_NAME, radar_utils.ECHO_TOP_50DBZ_NAME,
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME,
    radar_utils.REFL_COLUMN_MAX_NAME, radar_utils.MESH_NAME,
    radar_utils.REFL_0CELSIUS_NAME, radar_utils.REFL_M10CELSIUS_NAME,
    radar_utils.REFL_M20CELSIUS_NAME, radar_utils.REFL_LOWEST_ALTITUDE_NAME,
    radar_utils.SHI_NAME, radar_utils.VIL_NAME]

# TODO(thunderhoser): Deal with dual-pol variables in GridRad and the fact that
# they might be missing.
DEFAULT_GRIDRAD_FIELD_NAMES = [
    radar_utils.REFL_NAME, radar_utils.SPECTRUM_WIDTH_NAME,
    radar_utils.VORTICITY_NAME, radar_utils.DIVERGENCE_NAME]

DEFAULT_GRIDRAD_HEIGHTS_M_ASL = numpy.linspace(1000, 10000, num=10, dtype=int)


def _centroids_latlng_to_rowcol(
        centroid_latitudes_deg, centroid_longitudes_deg, nw_grid_point_lat_deg,
        nw_grid_point_lng_deg, lat_spacing_deg, lng_spacing_deg):
    """Converts storm centroids from lat-long to row-column coordinates.

    N = number of storm objects

    :param centroid_latitudes_deg: length-N numpy array with latitudes (deg N)
        of storm centroids.
    :param centroid_longitudes_deg: length-N numpy array with longitudes (deg E)
        of storm centroids.
    :param nw_grid_point_lat_deg: Latitude (deg N) of northwesternmost grid
        point.
    :param nw_grid_point_lng_deg: Longitude (deg E) of northwesternmost grid
        point.
    :param lat_spacing_deg: Spacing (deg N) between adjacent grid rows.
    :param lng_spacing_deg: Spacing (deg E) between adjacent grid columns.
    :return: centroid_rows: length-N numpy array with row indices (half-
        integers) of storm centroids.
    :return: centroid_columns: length-N numpy array with column indices (half-
        integers) of storm centroids.
    """

    center_row_indices, center_column_indices = radar_utils.latlng_to_rowcol(
        latitudes_deg=centroid_latitudes_deg,
        longitudes_deg=centroid_longitudes_deg,
        nw_grid_point_lat_deg=nw_grid_point_lat_deg,
        nw_grid_point_lng_deg=nw_grid_point_lng_deg,
        lat_spacing_deg=lat_spacing_deg, lng_spacing_deg=lng_spacing_deg)

    return (rounder.round_to_half_integer(center_row_indices),
            rounder.round_to_half_integer(center_column_indices))


def _get_storm_image_coords(
        num_full_grid_rows, num_full_grid_columns, num_storm_image_rows,
        num_storm_image_columns, center_row, center_column):
    """Generates row-column coordinates for storm image.

    :param num_full_grid_rows: Number of rows in full grid.
    :param num_full_grid_columns: Number of columns in full grid.
    :param num_storm_image_rows: Number of rows in storm image (subgrid).
    :param num_storm_image_columns: Number of columns in storm image (subgrid).
    :param center_row: Row index (half-integer) at center of storm image.
    :param center_column: Column index (half-integer) at center of storm image.
    :return: storm_image_coord_dict: Dictionary with the following keys.
    storm_image_coord_dict['first_storm_image_row']: First row (integer) in
        storm image.
    storm_image_coord_dict['last_storm_image_row']: Last row (integer) in storm
        image.
    storm_image_coord_dict['first_storm_image_column']: First column (integer)
        in storm image.
    storm_image_coord_dict['last_storm_image_column']: Last column (integer) in
        storm image.
    storm_image_coord_dict['num_padding_rows_at_top']: Number of padding rows at
        top of storm image.  This will be non-zero iff the storm image runs out
        of bounds of the full image.
    storm_image_coord_dict['num_padding_rows_at_bottom']: Number of padding rows
        at bottom of storm image.
    storm_image_coord_dict['num_padding_columns_at_left']: Number of padding
        columns at left of storm image.
    storm_image_coord_dict['num_padding_columns_at_right']: Number of padding
        columns at right of storm image.
    """

    first_storm_image_row = int(numpy.ceil(
        center_row - num_storm_image_rows / 2))
    last_storm_image_row = int(numpy.floor(
        center_row + num_storm_image_rows / 2))
    first_storm_image_column = int(numpy.ceil(
        center_column - num_storm_image_columns / 2))
    last_storm_image_column = int(numpy.floor(
        center_column + num_storm_image_columns / 2))

    if first_storm_image_row < 0:
        num_padding_rows_at_top = 0 - first_storm_image_row
        first_storm_image_row = 0
    else:
        num_padding_rows_at_top = 0

    if last_storm_image_row > num_full_grid_rows - 1:
        num_padding_rows_at_bottom = last_storm_image_row - (
            num_full_grid_rows - 1)
        last_storm_image_row = num_full_grid_rows - 1
    else:
        num_padding_rows_at_bottom = 0

    if first_storm_image_column < 0:
        num_padding_columns_at_left = 0 - first_storm_image_column
        first_storm_image_column = 0
    else:
        num_padding_columns_at_left = 0

    if last_storm_image_column > num_full_grid_columns - 1:
        num_padding_columns_at_right = last_storm_image_column - (
            num_full_grid_columns - 1)
        last_storm_image_column = num_full_grid_columns - 1
    else:
        num_padding_columns_at_right = 0

    return {
        FIRST_STORM_ROW_KEY: first_storm_image_row,
        LAST_STORM_ROW_KEY: last_storm_image_row,
        FIRST_STORM_COLUMN_KEY: first_storm_image_column,
        LAST_STORM_COLUMN_KEY: last_storm_image_column,
        NUM_TOP_PADDING_ROWS_KEY: num_padding_rows_at_top,
        NUM_BOTTOM_PADDING_ROWS_KEY: num_padding_rows_at_bottom,
        NUM_LEFT_PADDING_COLS_KEY: num_padding_columns_at_left,
        NUM_RIGHT_PADDING_COLS_KEY: num_padding_columns_at_right
    }


def _check_storm_images(
        storm_image_matrix, storm_ids, unix_time_sec, radar_field_name,
        radar_height_m_asl):
    """Checks storm images (e.g., created by extract_storm_image) for errors.

    K = number of storm objects
    M = number of rows in each image
    N = number of columns in each image

    :param storm_image_matrix: K-by-M-by-N numpy array with image for each storm
        object.
    :param storm_ids: length-K list of storm IDs (strings).
    :param unix_time_sec: Valid time.
    :param radar_field_name: Name of radar field (string).
    :param radar_height_m_asl: Height (metres above sea level) of radar field.
    """

    error_checking.assert_is_string_list(storm_ids)
    error_checking.assert_is_numpy_array(
        numpy.array(storm_ids), num_dimensions=1)
    error_checking.assert_is_integer(unix_time_sec)

    radar_utils.check_field_name(radar_field_name)
    error_checking.assert_is_geq(radar_height_m_asl, 0)

    num_storm_objects = len(storm_ids)
    error_checking.assert_is_numpy_array(storm_image_matrix, num_dimensions=3)
    error_checking.assert_is_numpy_array(
        storm_image_matrix, exact_dimensions=numpy.array(
            [num_storm_objects, storm_image_matrix.shape[1],
             storm_image_matrix.shape[2]]))


def _check_storm_labels(
        storm_ids, unix_time_sec, storm_to_winds_table,
        storm_to_tornadoes_table):
    """Checks storm labels (target variables) for errors.

    :param storm_ids: 1-D list of storm IDs (strings).
    :param unix_time_sec: Valid time.
    :param storm_to_winds_table: pandas DataFrame created by
        `labels.label_wind_speed_for_classification`.  This may be None.
    :param storm_to_tornadoes_table: pandas DataFrame created by
        `labels.label_tornado_occurrence`.  This may be None.
    :return: relevant_storm_to_winds_table: Same as input, but containing only
        the given storm objects (with ID belonging to `storm_ids` and valid time
        of `unix_time_sec`).  If input is None, this is None.
    :return: relevant_storm_to_tornadoes_table: See above.
    :raises: ValueError: if `storm_to_winds_table` is not None and is missing a
        storm object.
    :raises: ValueError: if `storm_to_tornadoes_table` is not None and is
        missing a storm object.
    """

    error_checking.assert_is_string_list(storm_ids)
    error_checking.assert_is_numpy_array(
        numpy.array(storm_ids), num_dimensions=1)
    error_checking.assert_is_integer(unix_time_sec)

    num_storm_objects = len(storm_ids)

    if storm_to_winds_table is None:
        relevant_storm_to_winds_table = None
    else:
        labels.check_wind_speed_label_table(storm_to_winds_table)

        relevant_storm_to_winds_table = storm_to_winds_table.loc[
            storm_to_winds_table[tracking_utils.TIME_COLUMN] == unix_time_sec]
        relevant_storm_to_winds_table = relevant_storm_to_winds_table.loc[
            relevant_storm_to_winds_table[tracking_utils.STORM_ID_COLUMN].isin(
                storm_ids)]

        num_labeled_objects = len(relevant_storm_to_winds_table.index)
        if num_labeled_objects < num_storm_objects:
            time_string = time_conversion.unix_sec_to_string(
                unix_time_sec, TIME_FORMAT)
            error_string = (
                'Expected {0:d} storm objects in wind-linkage table for {1:s}.'
                '  Instead, got {2:d} storm objects.'
            ).format(num_storm_objects, time_string, num_labeled_objects)
            raise ValueError(error_string)

    if storm_to_tornadoes_table is None:
        relevant_storm_to_tornadoes_table = None
    else:
        labels.check_tornado_label_table(storm_to_tornadoes_table)

        relevant_storm_to_tornadoes_table = storm_to_tornadoes_table.loc[
            storm_to_tornadoes_table[tracking_utils.TIME_COLUMN] ==
            unix_time_sec]
        relevant_storm_to_tornadoes_table = (
            relevant_storm_to_tornadoes_table.loc[
                relevant_storm_to_tornadoes_table[
                    tracking_utils.STORM_ID_COLUMN].isin(storm_ids)])

        num_labeled_objects = len(relevant_storm_to_tornadoes_table.index)
        if num_labeled_objects < num_storm_objects:
            time_string = time_conversion.unix_sec_to_string(
                unix_time_sec, TIME_FORMAT)
            error_string = (
                'Expected {0:d} storm objects in tornado-linkage table for '
                '{1:s}.  Instead, got {2:d} storm objects.'
            ).format(num_storm_objects, time_string, num_labeled_objects)
            raise ValueError(error_string)

    return relevant_storm_to_winds_table, relevant_storm_to_tornadoes_table


def _find_many_files_one_spc_date(
        top_directory_name, start_time_unix_sec, end_time_unix_sec,
        spc_date_string, radar_source, radar_field_names,
        reflectivity_heights_m_asl=None, raise_error_if_missing=True):
    """Finds many files containing storm images with MYRORSS or MRMS data.

    N = number of time steps
    P = number of field/height pairs

    :param top_directory_name: Name of top-level directory with storm-image
        files.
    :param start_time_unix_sec: Start time.  This method will find all files
        with the given params from `start_time_unix_sec`...`end_time_unix_sec`.
    :param end_time_unix_sec: See above.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param radar_source: Data source (either "myrorss" or "mrms").
    :param radar_field_names: See doc for
        `extract_storm_images_myrorss_or_mrms`.
    :param reflectivity_heights_m_asl: See doc for
        `extract_storm_images_myrorss_or_mrms`.
    :param raise_error_if_missing: Boolean flag.  If True and no files are
        found, this method will error out.
    :return: image_file_name_matrix: N-by-P numpy array of file paths.
    :return: unix_times_sec: length-N numpy array of valid times.
    :return: field_name_by_pair: length-P list with names of radar fields.
    :return: height_by_pair_m_asl: length-P numpy array of radar heights (metres
        above sea level).
    :raises: ValueError: if no files are found and raise_error_if_missing =
        True.
    """

    field_name_by_pair, height_by_pair_m_asl = (
        myrorss_and_mrms_utils.fields_and_refl_heights_to_pairs(
            field_names=radar_field_names, data_source=radar_source,
            refl_heights_m_asl=reflectivity_heights_m_asl))

    num_field_height_pairs = len(field_name_by_pair)
    image_file_name_matrix = None
    unix_times_sec = None

    for j in range(num_field_height_pairs):
        if j == 0:
            this_file_pattern = (
                '{0:s}/{1:s}/{2:s}/{3:s}/{4:s}/{5:05d}_metres_asl/'
                'storm_images_{6:s}.p').format(
                    top_directory_name, radar_source, spc_date_string[:4],
                    spc_date_string, field_name_by_pair[j],
                    numpy.round(int(height_by_pair_m_asl[j])),
                    TIME_FORMAT_REGEX)

            these_file_names = glob.glob(this_file_pattern)
            image_file_names = []
            unix_times_sec = []

            for this_file_name in these_file_names:
                _, this_pathless_file_name = os.path.split(this_file_name)
                this_extensionless_file_name, _ = os.path.splitext(
                    this_pathless_file_name)

                this_time_string = this_extensionless_file_name.split('_')[-1]
                this_time_unix_sec = time_conversion.string_to_unix_sec(
                    this_time_string, TIME_FORMAT)

                if (start_time_unix_sec <= this_time_unix_sec <=
                        end_time_unix_sec):
                    image_file_names.append(this_file_name)
                    unix_times_sec.append(this_time_unix_sec)

            num_times = len(image_file_names)
            if num_times == 0:
                if raise_error_if_missing:
                    error_string = (
                        'Cannot find any files on SPC date "{0:s}" for "{1:s}" '
                        'at {2:d} metres ASL.').format(
                            spc_date_string, field_name_by_pair[j],
                            numpy.round(int(height_by_pair_m_asl[j])))
                    raise ValueError(error_string)

                return None, None, field_name_by_pair, height_by_pair_m_asl

            unix_times_sec = numpy.array(unix_times_sec, dtype=int)
            image_file_name_matrix = numpy.full(
                (num_times, num_field_height_pairs), '', dtype=object)
            image_file_name_matrix[:, j] = numpy.array(
                image_file_names, dtype=object)

        else:
            for i in range(len(unix_times_sec)):
                image_file_name_matrix[i, j] = find_storm_image_file(
                    top_directory_name=top_directory_name,
                    unix_time_sec=unix_times_sec[i],
                    spc_date_string=spc_date_string,
                    radar_source=radar_source,
                    radar_field_name=field_name_by_pair[j],
                    radar_height_m_asl=height_by_pair_m_asl[j],
                    raise_error_if_missing=True)

    return (image_file_name_matrix, unix_times_sec, field_name_by_pair,
            height_by_pair_m_asl)


def extract_storm_image(
        full_radar_matrix, center_row, center_column,
        num_storm_image_rows=DEFAULT_NUM_IMAGE_ROWS,
        num_storm_image_columns=DEFAULT_NUM_IMAGE_COLUMNS):
    """Extracts storm-centered radar image from full radar image.

    M = number of rows (unique grid-point latitudes) in full grid
    N = number of columns (unique grid-point longitudes) in full grid
    m = number of rows in storm image
    n = number of columns in storm image

    :param full_radar_matrix: M-by-N numpy array of radar values (one variable,
        one height, one time step).
    :param center_row: Row index (half-integer) at center of storm image.
    :param center_column: Column index (half-integer) at center of storm image.
    :param num_storm_image_rows: m, defined above.
    :param num_storm_image_columns: n, defined above.
    :return: storm_centered_radar_matrix: m-by-n numpy array of radar values
        (same variable, height, and time step).
    """

    error_checking.assert_is_real_numpy_array(full_radar_matrix)
    error_checking.assert_is_numpy_array(full_radar_matrix, num_dimensions=2)
    num_full_grid_rows = full_radar_matrix.shape[0]
    num_full_grid_columns = full_radar_matrix.shape[1]

    error_checking.assert_is_geq(center_row, -0.5)
    error_checking.assert_is_leq(center_row, num_full_grid_rows - 0.5)
    error_checking.assert_is_geq(center_column, -0.5)
    error_checking.assert_is_leq(center_column, num_full_grid_columns - 0.5)

    error_checking.assert_is_integer(num_storm_image_rows)
    error_checking.assert_is_geq(num_storm_image_rows, MIN_NUM_IMAGE_ROWS)
    error_checking.assert_is_integer(num_storm_image_columns)
    error_checking.assert_is_geq(num_storm_image_columns, MIN_NUM_IMAGE_COLUMNS)

    storm_image_coord_dict = _get_storm_image_coords(
        num_full_grid_rows=num_full_grid_rows,
        num_full_grid_columns=num_full_grid_columns,
        num_storm_image_rows=num_storm_image_rows,
        num_storm_image_columns=num_storm_image_columns, center_row=center_row,
        center_column=center_column)

    storm_image_rows = numpy.linspace(
        storm_image_coord_dict[FIRST_STORM_ROW_KEY],
        storm_image_coord_dict[LAST_STORM_ROW_KEY],
        num=(storm_image_coord_dict[LAST_STORM_ROW_KEY] -
             storm_image_coord_dict[FIRST_STORM_ROW_KEY] + 1),
        dtype=int)
    storm_image_columns = numpy.linspace(
        storm_image_coord_dict[FIRST_STORM_COLUMN_KEY],
        storm_image_coord_dict[LAST_STORM_COLUMN_KEY],
        num=(storm_image_coord_dict[LAST_STORM_COLUMN_KEY] -
             storm_image_coord_dict[FIRST_STORM_COLUMN_KEY] + 1),
        dtype=int)

    storm_centered_radar_matrix = numpy.take(
        full_radar_matrix, storm_image_rows, axis=0)
    storm_centered_radar_matrix = numpy.take(
        storm_centered_radar_matrix, storm_image_columns, axis=1)

    pad_width_input_arg = (
        (storm_image_coord_dict[NUM_TOP_PADDING_ROWS_KEY],
         storm_image_coord_dict[NUM_BOTTOM_PADDING_ROWS_KEY]),
        (storm_image_coord_dict[NUM_LEFT_PADDING_COLS_KEY],
         storm_image_coord_dict[NUM_RIGHT_PADDING_COLS_KEY]))

    return numpy.pad(
        storm_centered_radar_matrix, pad_width=pad_width_input_arg,
        mode='constant', constant_values=PADDING_VALUE)


def extract_storm_images_myrorss_or_mrms(
        storm_object_table, radar_source, top_radar_dir_name,
        top_output_dir_name, num_storm_image_rows=DEFAULT_NUM_IMAGE_ROWS,
        num_storm_image_columns=DEFAULT_NUM_IMAGE_COLUMNS,
        radar_field_names=DEFAULT_MYRORSS_MRMS_FIELD_NAMES,
        reflectivity_heights_m_asl=None):
    """Extracts storm-centered radar image for each field, height, storm object.

    K = number of storm objects
    F = number of radar fields
    P = number of field/height pairs
    T = number of time steps with storm objects

    :param storm_object_table: K-row pandas DataFrame with the following
        columns.
    storm_object_table.storm_id: String ID for storm cell.
    storm_object_table.unix_time_sec: Valid time.
    storm_object_table.spc_date_unix_sec: SPC date.
    storm_object_table.centroid_lat_deg: Latitude (deg N) of storm centroid.
    storm_object_table.centroid_lng_deg: Longitude (deg E) of storm centroid.

    :param radar_source: Data source (either "myrorss" or "mrms").
    :param top_radar_dir_name: [input] Name of top-level directory with radar
        data from the given source.
    :param top_output_dir_name: [output] Name of top-level output directory for
        storm-centered radar images.
    :param num_storm_image_rows: Number of rows (unique grid-point latitudes) in
        each storm image.
    :param num_storm_image_columns: Number of columns (unique grid-point
        longitudes) in each storm image.
    :param radar_field_names: length-F list with names of radar fields.
    :param reflectivity_heights_m_asl: 1-D numpy array of heights (metres above
        sea level) for one radar field ("reflectivity_dbz").  If
        "reflectivity_dbz" is not in the list `radar_field_names`, you can leave
        this as None.
    :return: image_file_name_matrix: T-by-P numpy array of paths to output
        files.
    """

    # Find radar files.
    spc_date_strings = [
        time_conversion.time_to_spc_date_string(t)
        for t in storm_object_table[tracking_utils.SPC_DATE_COLUMN].values]

    file_dictionary = myrorss_and_mrms_io.find_many_raw_files(
        desired_times_unix_sec=
        storm_object_table[tracking_utils.TIME_COLUMN].values.astype(int),
        spc_date_strings=spc_date_strings, data_source=radar_source,
        field_names=radar_field_names, top_directory_name=top_radar_dir_name,
        reflectivity_heights_m_asl=reflectivity_heights_m_asl)

    radar_file_name_matrix = file_dictionary[
        myrorss_and_mrms_io.RADAR_FILE_NAMES_KEY]
    valid_times_unix_sec = file_dictionary[myrorss_and_mrms_io.UNIQUE_TIMES_KEY]
    valid_spc_dates_unix_sec = file_dictionary[
        myrorss_and_mrms_io.SPC_DATES_AT_UNIQUE_TIMES_KEY]
    radar_field_name_by_pair = file_dictionary[
        myrorss_and_mrms_io.FIELD_NAME_BY_PAIR_KEY]
    radar_height_by_pair_m_asl = file_dictionary[
        myrorss_and_mrms_io.HEIGHT_BY_PAIR_KEY]

    valid_time_strings = [time_conversion.unix_sec_to_string(t, TIME_FORMAT)
                          for t in valid_times_unix_sec]
    spc_date_strings = [time_conversion.time_to_spc_date_string(t)
                        for t in valid_spc_dates_unix_sec]

    # Initialize output array.
    num_field_height_pairs = len(radar_field_name_by_pair)
    num_valid_times = len(valid_times_unix_sec)
    image_file_name_matrix = numpy.full(
        (num_valid_times, num_field_height_pairs), '', dtype=object)

    for i in range(num_valid_times):

        # Find storm objects at [i]th valid time.
        these_storm_flags = numpy.logical_and(
            storm_object_table[tracking_utils.TIME_COLUMN].values ==
            valid_times_unix_sec[i],
            storm_object_table[tracking_utils.SPC_DATE_COLUMN].values ==
            valid_spc_dates_unix_sec[i])

        these_storm_indices = numpy.where(these_storm_flags)[0]
        these_storm_ids = storm_object_table[
            tracking_utils.STORM_ID_COLUMN].values[these_storm_indices].tolist()
        this_num_storms = len(these_storm_indices)

        for j in range(num_field_height_pairs):
            if radar_file_name_matrix[i, j] is None:
                continue

            this_storm_image_matrix = numpy.full(
                (this_num_storms, num_storm_image_rows,
                 num_storm_image_columns), numpy.nan)

            print (
                'Extracting storm images for "{0:s}" at {1:d} metres ASL and '
                '{2:s}...').format(
                    radar_field_name_by_pair[j],
                    numpy.round(int(radar_height_by_pair_m_asl[j])),
                    valid_time_strings[i])

            # Read data for [j]th field/height pair at [i]th time step.
            this_metadata_dict = (
                myrorss_and_mrms_io.read_metadata_from_raw_file(
                    radar_file_name_matrix[i, j], data_source=radar_source))

            this_sparse_grid_table = (
                myrorss_and_mrms_io.read_data_from_sparse_grid_file(
                    radar_file_name_matrix[i, j],
                    field_name_orig=this_metadata_dict[
                        myrorss_and_mrms_io.FIELD_NAME_COLUMN_ORIG],
                    data_source=radar_source,
                    sentinel_values=this_metadata_dict[
                        radar_utils.SENTINEL_VALUE_COLUMN]))

            this_radar_matrix, _, _ = radar_s2f.sparse_to_full_grid(
                this_sparse_grid_table, this_metadata_dict)
            this_radar_matrix[numpy.isnan(this_radar_matrix)] = 0.

            # Extract storm images for [j]th field/height pair at [i]th time
            # step.
            these_center_rows, these_center_columns = (
                _centroids_latlng_to_rowcol(
                    centroid_latitudes_deg=storm_object_table[
                        tracking_utils.CENTROID_LAT_COLUMN].values[
                            these_storm_indices],
                    centroid_longitudes_deg=storm_object_table[
                        tracking_utils.CENTROID_LNG_COLUMN].values[
                            these_storm_indices],
                    nw_grid_point_lat_deg=this_metadata_dict[
                        radar_utils.NW_GRID_POINT_LAT_COLUMN],
                    nw_grid_point_lng_deg=this_metadata_dict[
                        radar_utils.NW_GRID_POINT_LNG_COLUMN],
                    lat_spacing_deg=this_metadata_dict[
                        radar_utils.LAT_SPACING_COLUMN],
                    lng_spacing_deg=this_metadata_dict[
                        radar_utils.LNG_SPACING_COLUMN]))

            for k in range(this_num_storms):
                this_storm_image_matrix[k, :, :] = extract_storm_image(
                    full_radar_matrix=this_radar_matrix,
                    center_row=these_center_rows[k],
                    center_column=these_center_columns[k],
                    num_storm_image_rows=num_storm_image_rows,
                    num_storm_image_columns=num_storm_image_columns)

            # Write storm images for [j]th field/height pair at [i]th time step.
            image_file_name_matrix[i, j] = find_storm_image_file(
                top_directory_name=top_output_dir_name,
                unix_time_sec=valid_times_unix_sec[i],
                spc_date_string=spc_date_strings[i], radar_source=radar_source,
                radar_field_name=radar_field_name_by_pair[j],
                radar_height_m_asl=radar_height_by_pair_m_asl[j],
                raise_error_if_missing=False)

            print 'Writing storm images to "{0:s}"...\n'.format(
                image_file_name_matrix[i, j])
            write_storm_images(
                pickle_file_name=image_file_name_matrix[i, j],
                storm_image_matrix=this_storm_image_matrix,
                storm_ids=these_storm_ids,
                unix_time_sec=valid_times_unix_sec[i],
                radar_field_name=radar_field_name_by_pair[j],
                radar_height_m_asl=radar_height_by_pair_m_asl[j])

    return image_file_name_matrix


def extract_storm_images_gridrad(
        storm_object_table, top_radar_dir_name, top_output_dir_name,
        num_storm_image_rows=DEFAULT_NUM_IMAGE_ROWS,
        num_storm_image_columns=DEFAULT_NUM_IMAGE_COLUMNS,
        radar_field_names=DEFAULT_GRIDRAD_FIELD_NAMES,
        radar_heights_m_asl=DEFAULT_GRIDRAD_HEIGHTS_M_ASL):
    """Extracts storm-centered radar image for each field, height, storm object.

    K = number of storm objects
    F = number of radar fields
    H = number of radar heights
    T = number of time steps with storm objects

    :param storm_object_table: See documentation for
        `extract_storm_images_myrorss_or_mrms`.
    :param top_radar_dir_name: See doc for
        `extract_storm_images_myrorss_or_mrms`.
    :param top_output_dir_name: See doc for
        `extract_storm_images_myrorss_or_mrms`.
    :param num_storm_image_rows: See doc for
        `extract_storm_images_myrorss_or_mrms`.
    :param num_storm_image_columns: See doc for
        `extract_storm_images_myrorss_or_mrms`.
    :param radar_field_names: length-F list with names of radar fields.
    :param radar_heights_m_asl: length-H numpy array of radar heights (metres
        above sea level).
    :return: image_file_name_matrix: T-by-F-by-H numpy array of paths to output
        files.
    """

    _, _ = gridrad_utils.fields_and_refl_heights_to_pairs(
        field_names=radar_field_names, heights_m_asl=radar_heights_m_asl)
    radar_heights_m_asl = numpy.sort(
        numpy.round(radar_heights_m_asl).astype(int))

    # Find radar files.
    valid_times_unix_sec = numpy.unique(
        storm_object_table[tracking_utils.TIME_COLUMN].values)
    valid_time_strings = [time_conversion.unix_sec_to_string(t, TIME_FORMAT)
                          for t in valid_times_unix_sec]

    num_radar_times = len(valid_times_unix_sec)
    radar_file_names = [None] * num_radar_times
    for i in range(num_radar_times):
        radar_file_names[i] = gridrad_io.find_file(
            unix_time_sec=valid_times_unix_sec[i],
            top_directory_name=top_radar_dir_name, raise_error_if_missing=True)

    # Initialize output array.
    num_fields = len(radar_field_names)
    num_heights = len(radar_heights_m_asl)
    num_valid_times = len(valid_times_unix_sec)
    image_file_name_matrix = numpy.full(
        (num_valid_times, num_fields, num_heights), '', dtype=object)

    for i in range(num_valid_times):

        # Read metadata for [i]th valid time.
        this_metadata_dict = gridrad_io.read_metadata_from_full_grid_file(
            radar_file_names[i])

        # Find storm objects at [i]th valid time.
        these_storm_indices = numpy.where(
            storm_object_table[tracking_utils.TIME_COLUMN].values ==
            valid_times_unix_sec[i])[0]
        these_storm_ids = storm_object_table[
            tracking_utils.STORM_ID_COLUMN].values[these_storm_indices].tolist()
        this_num_storms = len(these_storm_indices)

        for j in range(num_fields):

            # Read data for [j]th field at [i]th valid time.
            print 'Reading "{0:s}" from file: "{1:s}"...'.format(
                radar_field_names[j], radar_file_names[i])

            this_field_radar_matrix, these_grid_point_heights_m_asl, _, _ = (
                gridrad_io.read_field_from_full_grid_file(
                    radar_file_names[i], field_name=radar_field_names[j],
                    metadata_dict=this_metadata_dict))

            these_grid_point_heights_m_asl = numpy.round(
                these_grid_point_heights_m_asl).astype(int)
            these_height_indices_to_keep = numpy.array(
                [these_grid_point_heights_m_asl.tolist().index(h)
                 for h in radar_heights_m_asl], dtype=int)
            del these_grid_point_heights_m_asl

            this_field_radar_matrix = (
                this_field_radar_matrix[these_height_indices_to_keep, :, :])
            this_field_radar_matrix[numpy.isnan(this_field_radar_matrix)] = 0.

            for k in range(num_heights):
                this_storm_image_matrix = numpy.full(
                    (this_num_storms, num_storm_image_rows,
                     num_storm_image_columns), numpy.nan)

                print (
                    'Extracting storm images for "{0:s}" at {1:d} metres ASL '
                    'and {2:s}...').format(
                        radar_field_names[j],
                        numpy.round(int(radar_heights_m_asl[k])),
                        valid_time_strings[i])

                # Extract storm images for [j]th field at [k]th height and [i]th
                # time step.
                these_center_rows, these_center_columns = (
                    _centroids_latlng_to_rowcol(
                        centroid_latitudes_deg=storm_object_table[
                            tracking_utils.CENTROID_LAT_COLUMN].values[
                                these_storm_indices],
                        centroid_longitudes_deg=storm_object_table[
                            tracking_utils.CENTROID_LNG_COLUMN].values[
                                these_storm_indices],
                        nw_grid_point_lat_deg=this_metadata_dict[
                            radar_utils.NW_GRID_POINT_LAT_COLUMN],
                        nw_grid_point_lng_deg=this_metadata_dict[
                            radar_utils.NW_GRID_POINT_LNG_COLUMN],
                        lat_spacing_deg=this_metadata_dict[
                            radar_utils.LAT_SPACING_COLUMN],
                        lng_spacing_deg=this_metadata_dict[
                            radar_utils.LNG_SPACING_COLUMN]))

                for m in range(this_num_storms):
                    this_storm_image_matrix[m, :, :] = extract_storm_image(
                        full_radar_matrix=this_field_radar_matrix[k, ...],
                        center_row=these_center_rows[m],
                        center_column=these_center_columns[m],
                        num_storm_image_rows=num_storm_image_rows,
                        num_storm_image_columns=num_storm_image_columns)

                # Write storm images for [j]th field at [k]th height and [i]th
                # time step.
                image_file_name_matrix[i, j, k] = find_storm_image_file(
                    top_directory_name=top_output_dir_name,
                    unix_time_sec=valid_times_unix_sec[i],
                    spc_date_string=time_conversion.time_to_spc_date_string(
                        valid_times_unix_sec[i]),
                    radar_source=radar_utils.GRIDRAD_SOURCE_ID,
                    radar_field_name=radar_field_names[j],
                    radar_height_m_asl=radar_heights_m_asl[k],
                    raise_error_if_missing=False)

                print 'Writing storm images to "{0:s}"...\n'.format(
                    image_file_name_matrix[i, j, k])
                write_storm_images(
                    pickle_file_name=image_file_name_matrix[i, j, k],
                    storm_image_matrix=this_storm_image_matrix,
                    storm_ids=these_storm_ids,
                    unix_time_sec=valid_times_unix_sec[i],
                    radar_field_name=radar_field_names[j],
                    radar_height_m_asl=radar_heights_m_asl[k])

    return image_file_name_matrix


def find_storm_image_file(
        top_directory_name, unix_time_sec, spc_date_string, radar_source,
        radar_field_name, radar_height_m_asl, raise_error_if_missing=True):
    """Finds file with storm imgs (e.g., those created by extract_storm_image).

    :param top_directory_name: Name of top-level directory with storm-image
        files.
    :param unix_time_sec: Valid time.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param radar_source: Data source (must be accepted by
        `radar_utils.check_data_source`).
    :param radar_field_name: Name of radar field (string).
    :param radar_height_m_asl: Height (metres above sea level) of radar field.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will error out.
    :return: storm_image_file_name: Path to image file.  If file is missing and
        raise_error_if_missing = False, this is the *expected* path.
    :raises: ValueError: if file is missing and raise_error_if_missing = True.
    """

    error_checking.assert_is_string(top_directory_name)
    _ = time_conversion.spc_date_string_to_unix_sec(spc_date_string)
    radar_utils.check_data_source(radar_source)
    error_checking.assert_is_string(radar_field_name)
    error_checking.assert_is_greater(radar_height_m_asl, 0)
    error_checking.assert_is_boolean(raise_error_if_missing)

    storm_image_file_name = (
        '{0:s}/{1:s}/{2:s}/{3:s}/{4:s}/{5:05d}_metres_asl/'
        'storm_images_{6:s}.p').format(
            top_directory_name, radar_source, spc_date_string[:4],
            spc_date_string, radar_field_name,
            numpy.round(int(radar_height_m_asl)),
            time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT))

    if raise_error_if_missing and not os.path.isfile(storm_image_file_name):
        error_string = (
            'Cannot find file with storm-centered radar images.  Expected at: '
            '{0:s}').format(storm_image_file_name)
        raise ValueError(error_string)

    return storm_image_file_name


def find_many_files_myrorss_or_mrms(
        top_directory_name, start_time_unix_sec, end_time_unix_sec,
        radar_source, radar_field_names, reflectivity_heights_m_asl=None,
        raise_error_if_missing=True):
    """Finds many files containing storm images with MYRORSS or MRMS data.

    :param top_directory_name: See documentation for
        `find_many_files_one_spc_date`.
    :param start_time_unix_sec: Same.
    :param end_time_unix_sec: Same.
    :param radar_source: Same.
    :param radar_field_names: Same.
    :param reflectivity_heights_m_asl: Same.
    :param raise_error_if_missing: Same.
    :return: image_file_name_matrix: Same.
    :return: unix_times_sec: Same.
    :return: field_name_by_pair: Same.
    :return: height_by_pair_m_asl: Same.
    :raises: ValueError: if no files are found and raise_error_if_missing =
        True.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_integer(start_time_unix_sec)
    error_checking.assert_is_integer(end_time_unix_sec)
    error_checking.assert_is_greater(end_time_unix_sec, start_time_unix_sec)
    error_checking.assert_is_boolean(raise_error_if_missing)

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=time_conversion.time_to_spc_date_string(
            start_time_unix_sec),
        last_spc_date_string=time_conversion.time_to_spc_date_string(
            end_time_unix_sec))
    num_spc_dates = len(spc_date_strings)

    image_file_name_matrix = None
    unix_times_sec = None
    field_name_by_pair = None
    height_by_pair_m_asl = None

    for i in range(num_spc_dates):
        (this_file_name_matrix,
         these_times_unix_sec,
         field_name_by_pair,
         height_by_pair_m_asl) = _find_many_files_one_spc_date(
             top_directory_name=top_directory_name,
             start_time_unix_sec=start_time_unix_sec,
             end_time_unix_sec=end_time_unix_sec,
             spc_date_string=spc_date_strings[i], radar_source=radar_source,
             radar_field_names=radar_field_names,
             reflectivity_heights_m_asl=reflectivity_heights_m_asl,
             raise_error_if_missing=False)

        if this_file_name_matrix is None:
            continue

        if image_file_name_matrix is None:
            image_file_name_matrix = copy.deepcopy(this_file_name_matrix)
            unix_times_sec = copy.deepcopy(these_times_unix_sec)
        else:
            image_file_name_matrix = numpy.concatenate(
                (image_file_name_matrix, this_file_name_matrix), axis=0)
            unix_times_sec = numpy.concatenate((
                unix_times_sec, these_times_unix_sec))

    if raise_error_if_missing and image_file_name_matrix is None:
        start_time_string = time_conversion.unix_sec_to_string(
            start_time_unix_sec, TIME_FORMAT)
        end_time_string = time_conversion.unix_sec_to_string(
            end_time_unix_sec, TIME_FORMAT)
        error_string = 'Cannot find any files from {0:s} to {1:s}.'.format(
            start_time_string, end_time_string)
        raise ValueError(error_string)

    return (image_file_name_matrix, unix_times_sec, field_name_by_pair,
            height_by_pair_m_asl)


def find_many_files_gridrad(
        top_directory_name, start_time_unix_sec, end_time_unix_sec,
        radar_field_names, radar_heights_m_asl, raise_error_if_missing=True):
    """Finds many files containing storm images with GridRad data.

    N = number of time steps
    F = number of radar fields
    H = number of radar heights

    :param top_directory_name: Name of top-level directory with storm-image
        files.
    :param start_time_unix_sec: Start time.  This method will find all files
        with the given params from `start_time_unix_sec`...`end_time_unix_sec`.
    :param end_time_unix_sec: See above.
    :param radar_field_names: length-F list with names of radar fields.
    :param radar_heights_m_asl: length-H numpy array of radar heights (metres
        above sea level).
    :param raise_error_if_missing: Boolean flag.  If True and no files are
        found, this method will error out.
    :return: image_file_name_matrix: N-by-F-by-H numpy array of file paths.
    :return: unix_times_sec: length-N numpy array of valid times.
    :raises: ValueError: if no files are found and raise_error_if_missing =
        True.
    """

    _, _ = gridrad_utils.fields_and_refl_heights_to_pairs(
        field_names=radar_field_names, heights_m_asl=radar_heights_m_asl)

    all_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=start_time_unix_sec,
        end_time_unix_sec=end_time_unix_sec,
        time_interval_sec=GRIDRAD_TIME_INTERVAL_SEC, include_endpoint=True)

    image_file_name_matrix = None
    unix_times_sec = None
    num_fields = len(radar_field_names)
    num_heights = len(radar_heights_m_asl)

    for j in range(num_fields):
        for k in range(num_heights):
            if j == 0 and k == 0:
                image_file_names = []
                unix_times_sec = []

                for this_time_unix_sec in all_times_unix_sec:
                    this_file_name = find_storm_image_file(
                        top_directory_name=top_directory_name,
                        unix_time_sec=this_time_unix_sec,
                        spc_date_string=time_conversion.time_to_spc_date_string(
                            this_time_unix_sec),
                        radar_source=radar_utils.GRIDRAD_SOURCE_ID,
                        radar_field_name=radar_field_names[j],
                        radar_height_m_asl=radar_heights_m_asl[k],
                        raise_error_if_missing=False)

                    if not os.path.isfile(this_file_name):
                        continue

                    unix_times_sec.append(this_time_unix_sec)
                    image_file_names.append(this_file_name)

                num_times = len(image_file_names)
                if num_times == 0:
                    if raise_error_if_missing:
                        start_time_string = (
                            time_conversion.unix_sec_to_string(
                                start_time_unix_sec, TIME_FORMAT))
                        end_time_string = (
                            time_conversion.unix_sec_to_string(
                                end_time_unix_sec, TIME_FORMAT))
                        error_string = (
                            'Cannot find any files from {0:s} to {1:s}.'
                        ).format(start_time_string, end_time_string)
                        raise ValueError(error_string)

                    return None, None

                unix_times_sec = numpy.array(unix_times_sec, dtype=int)
                image_file_name_matrix = numpy.full(
                    (num_times, num_fields, num_heights), '', dtype=object)
                image_file_name_matrix[:, j, k] = numpy.array(
                    image_file_names, dtype=object)

            else:
                for i in range(len(unix_times_sec)):
                    image_file_name_matrix[i, j, k] = find_storm_image_file(
                        top_directory_name=top_directory_name,
                        unix_time_sec=unix_times_sec[i],
                        spc_date_string=time_conversion.time_to_spc_date_string(
                            unix_times_sec[i]),
                        radar_source=radar_utils.GRIDRAD_SOURCE_ID,
                        radar_field_name=radar_field_names[j],
                        radar_height_m_asl=radar_heights_m_asl[k],
                        raise_error_if_missing=True)

    return image_file_name_matrix, unix_times_sec


def write_storm_images(
        pickle_file_name, storm_image_matrix, storm_ids, unix_time_sec,
        radar_field_name, radar_height_m_asl, storm_to_winds_table=None,
        storm_to_tornadoes_table=None):
    """Writes storm imgs (e.g., created by extract_storm_image) to Pickle file.

    :param pickle_file_name: Path to output file.
    :param storm_image_matrix: See documentation for `_check_storm_images`.
    :param storm_ids: See doc for `_check_storm_images`.
    :param unix_time_sec: Valid time.
    :param radar_field_name: See doc for `_check_storm_images`.
    :param radar_height_m_asl: See doc for `_check_storm_images`.
    :param storm_to_winds_table: See doc for `_check_storm_labels`.
    :param storm_to_tornadoes_table: See doc for `_check_storm_labels`.
    """

    _check_storm_images(
        storm_image_matrix=storm_image_matrix, storm_ids=storm_ids,
        unix_time_sec=unix_time_sec, radar_field_name=radar_field_name,
        radar_height_m_asl=radar_height_m_asl)

    _check_storm_labels(
        storm_ids=storm_ids, unix_time_sec=unix_time_sec,
        storm_to_winds_table=storm_to_winds_table,
        storm_to_tornadoes_table=storm_to_tornadoes_table)

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(storm_image_matrix, pickle_file_handle)
    pickle.dump(storm_ids, pickle_file_handle)
    pickle.dump(unix_time_sec, pickle_file_handle)
    pickle.dump(radar_field_name, pickle_file_handle)
    pickle.dump(radar_height_m_asl, pickle_file_handle)
    pickle.dump(storm_to_winds_table, pickle_file_handle)
    pickle.dump(storm_to_tornadoes_table, pickle_file_handle)
    pickle_file_handle.close()


def read_storm_images(pickle_file_name):
    """Reads storm imgs (e.g., created by extract_storm_image) from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: storm_image_dict: Dictionary with the following keys.
    storm_image_dict['storm_image_matrix']: See documentation for
        `_check_storm_images`.
    storm_image_dict['storm_ids']: See doc for `_check_storm_images`.
    storm_image_dict['unix_time_sec']: Valid time.
    storm_image_dict['radar_field_name']: See doc for `_check_storm_images`.
    storm_image_dict['radar_height_m_asl']: See doc for `_check_storm_images`.
    storm_image_dict['storm_to_winds_table']: See doc for `_check_storm_labels`.
    storm_image_dict['storm_to_tornadoes_table']: See doc for
        `_check_storm_labels`.
    """

    # TODO(thunderhoser): This method should return a dictionary.

    pickle_file_handle = open(pickle_file_name, 'rb')
    storm_image_matrix = pickle.load(pickle_file_handle)
    storm_ids = pickle.load(pickle_file_handle)
    unix_time_sec = pickle.load(pickle_file_handle)
    radar_field_name = pickle.load(pickle_file_handle)
    radar_height_m_asl = pickle.load(pickle_file_handle)
    storm_to_winds_table = pickle.load(pickle_file_handle)
    storm_to_tornadoes_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    _check_storm_images(
        storm_image_matrix=storm_image_matrix, storm_ids=storm_ids,
        unix_time_sec=unix_time_sec, radar_field_name=radar_field_name,
        radar_height_m_asl=radar_height_m_asl)

    _check_storm_labels(
        storm_ids=storm_ids, unix_time_sec=unix_time_sec,
        storm_to_winds_table=storm_to_winds_table,
        storm_to_tornadoes_table=storm_to_tornadoes_table)

    return {
        STORM_IMAGE_MATRIX_KEY: storm_image_matrix,
        STORM_IDS_KEY: storm_ids,
        VALID_TIME_KEY: unix_time_sec,
        RADAR_FIELD_NAME_KEY: radar_field_name,
        RADAR_HEIGHT_KEY: radar_height_m_asl,
        STORM_TO_WINDS_TABLE_KEY: storm_to_winds_table,
        STORM_TO_TORNADOES_TABLE_KEY: storm_to_tornadoes_table
    }


def extract_storm_labels(
        storm_ids, unix_time_sec, storm_to_winds_table=None,
        storm_to_tornadoes_table=None):
    """Extracts all labels (target variables) for each storm object.

    :param storm_ids: See documentation for `_check_storm_labels`.
    :param unix_time_sec: Same.
    :param storm_to_winds_table: Same.
    :param storm_to_tornadoes_table: Same.
    :return: relevant_storm_to_winds_table: Same.
    :return: relevant_storm_to_tornadoes_table: Same.
    :raises: ValueError: if `storm_to_winds_table` and
        `storm_to_tornadoes_table` are both None.
    """

    if storm_to_winds_table is None and storm_to_tornadoes_table is None:
        raise ValueError(
            'At least one of storm_to_winds_table and storm_to_tornadoes_table '
            'must be given (cannot both be None).')

    return _check_storm_labels(
        storm_ids=storm_ids, unix_time_sec=unix_time_sec,
        storm_to_winds_table=storm_to_winds_table,
        storm_to_tornadoes_table=storm_to_tornadoes_table)


def extract_one_label_per_storm(
        storm_ids, label_name, storm_to_winds_table=None,
        storm_to_tornadoes_table=None):
    """Extracts one label (target variable) for each storm object.

    K = number of storm objects

    :param storm_ids: length-K list of storm IDs (strings).
    :param label_name: Name of target variable.  This must be a column in
        `storm_to_winds_table` or `storm_to_tornadoes_table`.
    :param storm_to_winds_table: pandas DataFrame created by
        `extract_storm_labels`.
    :param storm_to_tornadoes_table: pandas DataFrame created by
        `extract_storm_labels`.
    :return: label_values: length-K numpy array of integers.  label_values[k] is
        the class for the [k]th storm object.
    """

    label_parameter_dict = labels.column_name_to_label_params(label_name)
    event_type_string = label_parameter_dict[labels.EVENT_TYPE_KEY]

    if event_type_string == events2storms.TORNADO_EVENT_TYPE_STRING:
        dummy_time_unix_sec = storm_to_tornadoes_table[
            tracking_utils.TIME_COLUMN].values[0]
        storm_ids_in_table = storm_to_tornadoes_table[
            tracking_utils.STORM_ID_COLUMN].values.tolist()
    else:
        dummy_time_unix_sec = storm_to_winds_table[
            tracking_utils.TIME_COLUMN].values[0]
        storm_ids_in_table = storm_to_winds_table[
            tracking_utils.STORM_ID_COLUMN].values.tolist()

    _check_storm_labels(
        storm_ids=storm_ids, unix_time_sec=dummy_time_unix_sec,
        storm_to_winds_table=storm_to_winds_table,
        storm_to_tornadoes_table=storm_to_tornadoes_table)

    sort_indices = numpy.array([storm_ids_in_table.index(s) for s in storm_ids])

    if event_type_string == events2storms.TORNADO_EVENT_TYPE_STRING:
        return storm_to_tornadoes_table[label_name].values[sort_indices]
    return storm_to_winds_table[label_name].values[sort_indices]


def attach_labels_to_storm_images(
        image_file_name_matrix, storm_to_winds_table=None,
        storm_to_tornadoes_table=None):
    """Attaches labels (target variables) to each storm-centered radar image.

    N = number of time steps
    P = number of radar variables (field/height pairs)

    :param image_file_name_matrix: N-by-P numpy array of paths to image files.
    :param storm_to_winds_table: pandas DataFrame created by
        `labels.label_wind_speed_for_classification`.  This may be None.
    :param storm_to_tornadoes_table: pandas DataFrame created by
        `labels.label_tornado_occurrence`.  This may be None.
    :raises: ValueError: if `storm_to_winds_table` and
        `storm_to_tornadoes_table` are both None.
    """

    if storm_to_winds_table is None and storm_to_tornadoes_table is None:
        raise ValueError(
            'At least one of storm_to_winds_table and storm_to_tornadoes_table '
            'must be given (cannot both be None).')

    error_checking.assert_is_numpy_array(
        image_file_name_matrix, num_dimensions=2)

    num_times = image_file_name_matrix.shape[0]
    num_field_height_pairs = image_file_name_matrix.shape[1]

    for i in range(num_times):
        for j in range(num_field_height_pairs):
            print 'Reading storm-centered radar images from: "{0:s}"...'.format(
                image_file_name_matrix[i, j])

            this_storm_image_dict = read_storm_images(
                image_file_name_matrix[i, j])

            print 'Attaching labels to storm images...'
            this_storm_to_winds_table, this_storm_to_tornadoes_table = (
                extract_storm_labels(
                    storm_ids=this_storm_image_dict[STORM_IDS_KEY],
                    unix_time_sec=this_storm_image_dict[VALID_TIME_KEY],
                    storm_to_winds_table=storm_to_winds_table,
                    storm_to_tornadoes_table=storm_to_tornadoes_table))

            print (
                'Writing storm images and labels back to: "{0:s}"...\n'
            ).format(image_file_name_matrix[i, j])
            write_storm_images(
                pickle_file_name=image_file_name_matrix[i, j],
                storm_image_matrix=
                this_storm_image_dict[STORM_IMAGE_MATRIX_KEY],
                storm_ids=this_storm_image_dict[STORM_IDS_KEY],
                unix_time_sec=this_storm_image_dict[VALID_TIME_KEY],
                radar_field_name=this_storm_image_dict[RADAR_FIELD_NAME_KEY],
                radar_height_m_asl=this_storm_image_dict[RADAR_HEIGHT_KEY],
                storm_to_winds_table=this_storm_to_winds_table,
                storm_to_tornadoes_table=this_storm_to_tornadoes_table)
