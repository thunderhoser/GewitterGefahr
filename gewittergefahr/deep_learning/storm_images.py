"""Methods for handling storm images.

A "storm image" is a radar image that shares a center with the storm object (in
other words, the center of the image is the centroid of the storm object).
"""

import os
import copy
import glob
import pickle
import numpy
import netCDF4
from gewittergefahr.gg_io import netcdf_io
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
GRID_SPACING_TOLERANCE_DEG = 1e-4
AZ_SHEAR_GRID_SPACING_MULTIPLIER = 2

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
VALID_TIMES_KEY = 'valid_times_unix_sec'
RADAR_FIELD_NAME_KEY = 'radar_field_name'
RADAR_HEIGHT_KEY = 'radar_height_m_asl'
STORM_TO_WINDS_TABLE_KEY = 'storm_to_winds_table'
STORM_TO_TORNADOES_TABLE_KEY = 'storm_to_tornadoes_table'
LABEL_VALUES_KEY = 'label_values'

LAT_DIMENSION_KEY = 'latitude'
LNG_DIMENSION_KEY = 'longitude'
CHARACTER_DIMENSION_KEY = 'storm_id_character'
STORM_OBJECT_DIMENSION_KEY = 'storm_object'

DEFAULT_NUM_IMAGE_ROWS = 32
DEFAULT_NUM_IMAGE_COLUMNS = 32
# MIN_NUM_IMAGE_ROWS = 8
# MIN_NUM_IMAGE_COLUMNS = 8
MIN_NUM_IMAGE_ROWS = 2
MIN_NUM_IMAGE_COLUMNS = 2

DEFAULT_RADAR_HEIGHTS_M_ASL = numpy.linspace(1000, 12000, num=12, dtype=int)

DEFAULT_MYRORSS_MRMS_FIELD_NAMES = [
    radar_utils.ECHO_TOP_18DBZ_NAME, radar_utils.ECHO_TOP_50DBZ_NAME,
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME,
    radar_utils.REFL_NAME, radar_utils.REFL_COLUMN_MAX_NAME,
    radar_utils.REFL_0CELSIUS_NAME, radar_utils.REFL_M10CELSIUS_NAME,
    radar_utils.REFL_M20CELSIUS_NAME, radar_utils.REFL_LOWEST_ALTITUDE_NAME,
    radar_utils.MESH_NAME, radar_utils.SHI_NAME, radar_utils.VIL_NAME]
AZIMUTHAL_SHEAR_FIELD_NAMES = [
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME]

DEFAULT_GRIDRAD_FIELD_NAMES = [
    radar_utils.REFL_NAME, radar_utils.SPECTRUM_WIDTH_NAME,
    radar_utils.VORTICITY_NAME, radar_utils.DIVERGENCE_NAME]


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
        storm_image_matrix, storm_ids, valid_times_unix_sec, radar_field_name,
        radar_height_m_asl):
    """Checks storm images (e.g., created by extract_storm_image) for errors.

    L = number of storm objects
    M = number of rows in each image
    N = number of columns in each image

    :param storm_image_matrix: L-by-M-by-N numpy array with image for each storm
        object.
    :param storm_ids: length-L list of storm IDs (strings).
    :param valid_times_unix_sec: length-L numpy array of valid times.
    :param radar_field_name: Name of radar field (string).
    :param radar_height_m_asl: Height (metres above sea level) of radar field.
    """

    error_checking.assert_is_string_list(storm_ids)
    error_checking.assert_is_numpy_array(
        numpy.array(storm_ids), num_dimensions=1)
    num_storm_objects = len(storm_ids)

    error_checking.assert_is_integer_numpy_array(valid_times_unix_sec)
    error_checking.assert_is_numpy_array(
        valid_times_unix_sec, exact_dimensions=numpy.array([num_storm_objects]))

    radar_utils.check_field_name(radar_field_name)
    error_checking.assert_is_geq(radar_height_m_asl, 0)

    error_checking.assert_is_numpy_array(storm_image_matrix, num_dimensions=3)
    error_checking.assert_is_numpy_array(
        storm_image_matrix, exact_dimensions=numpy.array(
            [num_storm_objects, storm_image_matrix.shape[1],
             storm_image_matrix.shape[2]]))


def _check_storm_labels(
        storm_ids, valid_times_unix_sec, storm_to_winds_table,
        storm_to_tornadoes_table):
    """Checks storm labels (target variables) for errors.

    L = number of storm objects

    :param storm_ids: length-L list of storm IDs (strings).
    :param valid_times_unix_sec: length-L numpy array of valid times.
    :param storm_to_winds_table: pandas DataFrame created by
        `labels.label_wind_speed_for_classification`.  This may be None.
    :param storm_to_tornadoes_table: pandas DataFrame created by
        `labels.label_tornado_occurrence`.  This may be None.
    :return: relevant_storm_to_winds_table: Same as input, but containing only
        the given storm objects (ID-time pairs).  If `storm_to_winds_table is
        None`, this is None.
    :return: relevant_storm_to_tornadoes_table: Same as input, but containing
        only the given storm objects (ID-time pairs).  If
        `storm_to_tornadoes_table is None`, this is None.
    """

    error_checking.assert_is_string_list(storm_ids)
    error_checking.assert_is_numpy_array(
        numpy.array(storm_ids), num_dimensions=1)
    num_storm_objects = len(storm_ids)

    error_checking.assert_is_integer_numpy_array(valid_times_unix_sec)
    error_checking.assert_is_numpy_array(
        valid_times_unix_sec, exact_dimensions=numpy.array([num_storm_objects]))

    if storm_to_winds_table is None:
        relevant_storm_to_winds_table = None
    else:
        labels.check_wind_speed_label_table(storm_to_winds_table)

        relevant_indices = _find_storm_objects(
            all_storm_ids=storm_to_winds_table[
                tracking_utils.STORM_ID_COLUMN].values,
            all_valid_times_unix_sec=storm_to_winds_table[
                tracking_utils.TIME_COLUMN].values,
            storm_ids_to_keep=storm_ids,
            valid_times_to_keep_unix_sec=valid_times_unix_sec)
        relevant_storm_to_winds_table = storm_to_winds_table.iloc[
            relevant_indices]

    if storm_to_tornadoes_table is None:
        relevant_storm_to_tornadoes_table = None
    else:
        labels.check_tornado_label_table(storm_to_tornadoes_table)

        relevant_indices = _find_storm_objects(
            all_storm_ids=storm_to_tornadoes_table[
                tracking_utils.STORM_ID_COLUMN].values,
            all_valid_times_unix_sec=storm_to_tornadoes_table[
                tracking_utils.TIME_COLUMN].values,
            storm_ids_to_keep=storm_ids,
            valid_times_to_keep_unix_sec=valid_times_unix_sec)
        relevant_storm_to_tornadoes_table = storm_to_tornadoes_table.iloc[
            relevant_indices]

    return relevant_storm_to_winds_table, relevant_storm_to_tornadoes_table


def _find_many_files_one_spc_date(
        top_directory_name, start_time_unix_sec, end_time_unix_sec,
        spc_date_string, radar_source, field_name_by_pair, height_by_pair_m_asl,
        raise_error_if_all_missing=True, raise_error_if_any_missing=False):
    """Finds many files containing storm images with MYRORSS or MRMS data.

    These files contain one time step each, all from the same SPC date.

    T = number of time steps
    C = number of field/height pairs

    :param top_directory_name: See documentation for
        `find_many_files_myrorss_or_mrms`.
    :param start_time_unix_sec: Start time.  This method will find all files
        from `start_time_unix_sec`...`end_time_unix_sec` that fit into the given
        SPC date.
    :param end_time_unix_sec: See above.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param radar_source: Source (either "myrorss" or "mrms").
    :param field_name_by_pair: length-C list with names of radar fields.
    :param height_by_pair_m_asl: length-C numpy array with heights (metres above
        sea level) of radar fields.
    :param raise_error_if_all_missing: Same.
    :param raise_error_if_any_missing: Same.
    :return: image_file_name_matrix: T-by-C numpy array of file paths.
    :return: valid_times_unix_sec: length-T numpy array of valid times.
    """

    # TODO(thunderhoser): Maybe this method should glob for all radar
    # field/height pairs, instead of just one.  Globbing for only one
    # field/height causes some files to be missed.

    num_field_height_pairs = len(field_name_by_pair)
    image_file_name_matrix = None
    valid_times_unix_sec = None

    is_field_az_shear = numpy.array(
        [this_field_name in AZIMUTHAL_SHEAR_FIELD_NAMES
         for this_field_name in field_name_by_pair])
    is_any_field_az_shear = numpy.any(is_field_az_shear)
    glob_index = -1

    for j in range(num_field_height_pairs):
        glob_now = (field_name_by_pair[j] in AZIMUTHAL_SHEAR_FIELD_NAMES
                    or not is_any_field_az_shear)
        if not glob_now:
            continue
        glob_index = j + 0

        this_file_pattern = (
            '{0:s}/{1:s}/{2:s}/{3:s}/{4:s}/{5:05d}_metres_asl/'
            'storm_images_{6:s}.nc'
        ).format(
            top_directory_name, radar_source, spc_date_string[:4],
            spc_date_string, field_name_by_pair[j],
            numpy.round(int(height_by_pair_m_asl[j])),
            TIME_FORMAT_REGEX)

        these_file_names = glob.glob(this_file_pattern)
        image_file_names = []
        valid_times_unix_sec = []

        for this_file_name in these_file_names:
            _, this_pathless_file_name = os.path.split(this_file_name)
            this_extensionless_file_name, _ = os.path.splitext(
                this_pathless_file_name)

            this_time_string = this_extensionless_file_name.split('_')[-1]
            this_time_unix_sec = time_conversion.string_to_unix_sec(
                this_time_string, TIME_FORMAT)

            if start_time_unix_sec <= this_time_unix_sec <= end_time_unix_sec:
                image_file_names.append(this_file_name)
                valid_times_unix_sec.append(this_time_unix_sec)

        num_times = len(image_file_names)
        if num_times == 0:
            if raise_error_if_all_missing:
                error_string = (
                    'Cannot find any files on SPC date "{0:s}" for "{1:s}" '
                    'at {2:d} metres ASL.'
                ).format(
                    spc_date_string, field_name_by_pair[j],
                    numpy.round(int(height_by_pair_m_asl[j])))
                raise ValueError(error_string)

            return None, None

        valid_times_unix_sec = numpy.array(valid_times_unix_sec, dtype=int)
        image_file_name_matrix = numpy.full(
            (num_times, num_field_height_pairs), '', dtype=object)
        image_file_name_matrix[:, j] = numpy.array(
            image_file_names, dtype=object)

        break

    for j in range(num_field_height_pairs):
        if j == glob_index:
            continue

        for i in range(len(valid_times_unix_sec)):
            image_file_name_matrix[i, j] = find_storm_image_file(
                top_directory_name=top_directory_name,
                unix_time_sec=valid_times_unix_sec[i],
                spc_date_string=spc_date_string, radar_source=radar_source,
                radar_field_name=field_name_by_pair[j],
                radar_height_m_asl=height_by_pair_m_asl[j],
                raise_error_if_missing=raise_error_if_any_missing)

            if not os.path.isfile(image_file_name_matrix[i, j]):
                image_file_name_matrix[i, j] = ''

    return image_file_name_matrix, valid_times_unix_sec


def _write_storm_images_only(
        netcdf_file_name, storm_image_matrix, storm_ids, valid_times_unix_sec,
        radar_field_name, radar_height_m_asl):
    """Writes storm-centered radar images to NetCDF file.

    These images should be created by `extract_storm_image`.

    :param netcdf_file_name: Path to output file.
    :param storm_image_matrix: See documentation for `_check_storm_images`.
    :param storm_ids: Same.
    :param valid_times_unix_sec: Same.
    :param radar_field_name: Same.
    :param radar_height_m_asl: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    netcdf_dataset = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    netcdf_dataset.setncattr(RADAR_FIELD_NAME_KEY, radar_field_name)
    netcdf_dataset.setncattr(RADAR_HEIGHT_KEY, radar_height_m_asl)

    num_storm_objects = storm_image_matrix.shape[0]
    num_storm_id_chars = 0
    for i in range(num_storm_objects):
        num_storm_id_chars = max([num_storm_id_chars, len(storm_ids[i])])

    netcdf_dataset.createDimension(
        STORM_OBJECT_DIMENSION_KEY, num_storm_objects)
    netcdf_dataset.createDimension(
        LAT_DIMENSION_KEY, storm_image_matrix.shape[1])
    netcdf_dataset.createDimension(
        LNG_DIMENSION_KEY, storm_image_matrix.shape[2])
    netcdf_dataset.createDimension(CHARACTER_DIMENSION_KEY, num_storm_id_chars)

    netcdf_dataset.createVariable(
        STORM_IDS_KEY, datatype='S1',
        dimensions=(STORM_OBJECT_DIMENSION_KEY, CHARACTER_DIMENSION_KEY))

    string_type = 'S{0:d}'.format(num_storm_id_chars)
    storm_ids_as_char_array = netCDF4.stringtochar(numpy.array(
        storm_ids, dtype=string_type))
    netcdf_dataset.variables[STORM_IDS_KEY][:] = numpy.array(
        storm_ids_as_char_array)

    netcdf_dataset.createVariable(
        VALID_TIMES_KEY, datatype=numpy.int32,
        dimensions=STORM_OBJECT_DIMENSION_KEY)
    netcdf_dataset.variables[VALID_TIMES_KEY][:] = valid_times_unix_sec

    chunk_size_tuple = (1,) + storm_image_matrix.shape[1:]
    netcdf_dataset.createVariable(
        STORM_IMAGE_MATRIX_KEY, datatype=numpy.float32,
        dimensions=(
            STORM_OBJECT_DIMENSION_KEY, LAT_DIMENSION_KEY, LNG_DIMENSION_KEY),
        chunksizes=chunk_size_tuple)

    netcdf_dataset.variables[STORM_IMAGE_MATRIX_KEY][:] = storm_image_matrix
    netcdf_dataset.close()


def _write_storm_labels_only(
        pickle_file_name, storm_to_winds_table=None,
        storm_to_tornadoes_table=None):
    """Writes storm-hazard labels to Pickle file.

    The input tables should be created by `extract_storm_labels`.

    :param pickle_file_name: Path to output file.
    :param storm_to_winds_table: See doc for `_check_storm_labels`.
    :param storm_to_tornadoes_table: See doc for `_check_storm_labels`.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(storm_to_winds_table, pickle_file_handle)
    pickle.dump(storm_to_tornadoes_table, pickle_file_handle)
    pickle_file_handle.close()


def _read_storm_labels_only(pickle_file_name):
    """Reads storm-hazard labels from Pickle file.

    This file should be written by `_write_storm_labels_only`.

    :param pickle_file_name: Path to input file.
    :return: storm_to_winds_table: See doc for `_check_storm_labels`.
    :return: storm_to_tornadoes_table: See doc for `_check_storm_labels`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    storm_to_winds_table = pickle.load(pickle_file_handle)
    storm_to_tornadoes_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    return storm_to_winds_table, storm_to_tornadoes_table


def _filter_storm_objects_by_label(
        label_values, num_storm_objects_class_dict, test_mode=False):
    """Filters storm objects by label (target variable).

    L = number of storm objects
    K_x = number of extended classes

    :param label_values: length-L numpy array of integers.  label_values[i] is
        the class for the [i]th storm object.
    :param num_storm_objects_class_dict: Dictionary, where each key is a class
        integer (-2 for dead storms) and each value is the corresponding number
        of storm objects to return.
    :param test_mode: Boolean flag.  Leave this False.
    :return: indices_to_keep: 1-D numpy array with indices of storm objects to
        keep.
    """

    indices_to_keep = numpy.array([], dtype=int)
    for this_class_integer in num_storm_objects_class_dict.keys():
        this_num_storm_objects = num_storm_objects_class_dict[
            this_class_integer]
        if this_num_storm_objects == 0:
            continue

        these_indices = numpy.where(label_values == this_class_integer)[0]
        if test_mode:
            these_indices = these_indices[:this_num_storm_objects]
        else:
            these_indices = numpy.random.choice(
                these_indices, size=this_num_storm_objects, replace=False)

        indices_to_keep = numpy.concatenate((indices_to_keep, these_indices))

    return indices_to_keep


def _find_storm_objects(
        all_storm_ids, all_valid_times_unix_sec, storm_ids_to_keep,
        valid_times_to_keep_unix_sec):
    """Finds storm objects.

    P = total number of storm objects
    p = number of storm objects to keep

    :param all_storm_ids: length-P list of storm IDs (strings).
    :param all_valid_times_unix_sec: length-P list of valid times.
    :param storm_ids_to_keep: length-p list of storm IDs (strings).
    :param valid_times_to_keep_unix_sec: length-p list of valid times.
    :return: relevant_indices: length-p numpy array with indices desired storm
        objects in the large arrays.
    :raises: ValueError: if any storm object (pair of storm ID and valid time)
        is non-unique.
    """

    num_storm_objects_total = len(all_storm_ids)
    num_storm_objects_to_keep = len(storm_ids_to_keep)

    all_storm_object_ids = [
        '{0:s}_{1:d}'.format(all_storm_ids[i], all_valid_times_unix_sec[i])
        for i in range(num_storm_objects_total)]
    storm_object_ids_to_keep = [
        '{0:s}_{1:d}'.format(storm_ids_to_keep[i],
                             valid_times_to_keep_unix_sec[i])
        for i in range(num_storm_objects_to_keep)]

    this_num_unique = len(set(all_storm_object_ids))
    if this_num_unique != len(all_storm_object_ids):
        error_string = (
            'Only {0:d} of {1:d} original storm objects are unique.'
        ).format(this_num_unique, len(all_storm_object_ids))
        raise ValueError(error_string)

    this_num_unique = len(set(storm_object_ids_to_keep))
    if this_num_unique != len(storm_object_ids_to_keep):
        error_string = (
            'Only {0:d} of {1:d} desired storm objects are unique.'
        ).format(this_num_unique, len(storm_object_ids_to_keep))
        raise ValueError(error_string)

    return numpy.array(
        [all_storm_object_ids.index(s) for s in storm_object_ids_to_keep],
        dtype=int)


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
        top_output_dir_name, one_file_per_time_step=False,
        num_storm_image_rows=DEFAULT_NUM_IMAGE_ROWS,
        num_storm_image_columns=DEFAULT_NUM_IMAGE_COLUMNS,
        radar_field_names=DEFAULT_MYRORSS_MRMS_FIELD_NAMES,
        reflectivity_heights_m_asl=DEFAULT_RADAR_HEIGHTS_M_ASL):
    """Extracts storm-centered radar image for each field, height, storm object.

    L = number of storm objects
    F = number of radar fields
    P = number of field/height pairs
    T = number of time steps with storm objects

    :param storm_object_table: L-row pandas DataFrame with the following
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
    :param one_file_per_time_step: Boolean flag.  If True, this method will
        write one file per radar field/height and time step.  If False, will
        write one file per radar field/height and SPC date.
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
    :raises: ValueError: if grid spacing is not uniform across all files.
    """

    error_checking.assert_is_boolean(one_file_per_time_step)

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
    unique_spc_dates_unix_sec = numpy.unique(valid_spc_dates_unix_sec)
    unique_spc_date_strings = [time_conversion.time_to_spc_date_string(t)
                               for t in unique_spc_dates_unix_sec]

    # Initialize output array.
    num_times = len(valid_time_strings)
    num_unique_spc_dates = len(unique_spc_date_strings)
    num_field_height_pairs = len(radar_field_name_by_pair)

    if one_file_per_time_step:
        image_file_name_matrix = numpy.full(
            (num_times, num_field_height_pairs), '', dtype=object)
    else:
        image_file_name_matrix = numpy.full(
            (num_unique_spc_dates, num_field_height_pairs), '', dtype=object)

    latitude_spacing_deg = None
    longitude_spacing_deg = None

    for q in range(num_unique_spc_dates):
        these_time_indices = numpy.where(
            valid_spc_dates_unix_sec == unique_spc_dates_unix_sec[q])[0]

        for j in range(num_field_height_pairs):
            this_date_storm_image_matrix = None
            this_date_storm_ids = []
            this_date_valid_times_unix_sec = numpy.array([], dtype=int)

            for i in these_time_indices:
                if radar_file_name_matrix[i, j] is None:
                    continue

                # Find storm objects at [i]th valid time.
                these_storm_flags = numpy.logical_and(
                    storm_object_table[tracking_utils.TIME_COLUMN].values ==
                    valid_times_unix_sec[i],
                    storm_object_table[tracking_utils.SPC_DATE_COLUMN].values ==
                    valid_spc_dates_unix_sec[i])

                these_storm_indices = numpy.where(these_storm_flags)[0]
                this_num_storms = len(these_storm_indices)

                these_storm_ids = storm_object_table[
                    tracking_utils.STORM_ID_COLUMN
                ].values[these_storm_indices].tolist()
                these_times_unix_sec = storm_object_table[
                    tracking_utils.TIME_COLUMN
                ].values[these_storm_indices].astype(int)

                if not one_file_per_time_step:
                    this_date_storm_ids += these_storm_ids
                    this_date_valid_times_unix_sec = numpy.concatenate((
                        this_date_valid_times_unix_sec, these_times_unix_sec))

                print (
                    'Extracting storm images for "{0:s}" at {1:d} metres ASL '
                    'and {2:s}...'
                ).format(
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

                if radar_field_name_by_pair[j] in AZIMUTHAL_SHEAR_FIELD_NAMES:
                    this_num_image_rows = (
                        num_storm_image_rows * AZ_SHEAR_GRID_SPACING_MULTIPLIER)
                    this_num_image_columns = (num_storm_image_columns *
                                              AZ_SHEAR_GRID_SPACING_MULTIPLIER)

                    this_lat_spacing_deg = (
                        AZ_SHEAR_GRID_SPACING_MULTIPLIER *
                        this_metadata_dict[radar_utils.LAT_SPACING_COLUMN])
                    this_lng_spacing_deg = (
                        AZ_SHEAR_GRID_SPACING_MULTIPLIER *
                        this_metadata_dict[radar_utils.LNG_SPACING_COLUMN])
                else:
                    this_num_image_rows = num_storm_image_rows + 0
                    this_num_image_columns = num_storm_image_columns + 0

                    this_lat_spacing_deg = this_metadata_dict[
                        radar_utils.LAT_SPACING_COLUMN]
                    this_lng_spacing_deg = this_metadata_dict[
                        radar_utils.LNG_SPACING_COLUMN]

                this_lat_spacing_deg = rounder.round_to_nearest(
                    this_lat_spacing_deg, GRID_SPACING_TOLERANCE_DEG)
                this_lng_spacing_deg = rounder.round_to_nearest(
                    this_lng_spacing_deg, GRID_SPACING_TOLERANCE_DEG)

                if latitude_spacing_deg is None:
                    latitude_spacing_deg = this_lat_spacing_deg + 0.
                    longitude_spacing_deg = this_lng_spacing_deg + 0.

                if (latitude_spacing_deg != this_lat_spacing_deg or
                        longitude_spacing_deg != this_lng_spacing_deg):
                    error_string = (
                        'First file has grid spacing of {0:.4f} deg lat, '
                        '{1:.4f} deg long.  This file ("{2:s}")  has grid '
                        'spacing of {3:.4f} deg lat, {4:.4f} deg long.  Grid '
                        'spacing should be uniform across all files.'
                    ).format(latitude_spacing_deg, longitude_spacing_deg,
                             radar_file_name_matrix[i, j], this_lat_spacing_deg,
                             this_lng_spacing_deg)
                    raise ValueError(error_string)

                this_storm_image_matrix = numpy.full(
                    (this_num_storms, this_num_image_rows,
                     this_num_image_columns), numpy.nan)

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
                        num_storm_image_rows=this_num_image_rows,
                        num_storm_image_columns=this_num_image_columns)

                if one_file_per_time_step:

                    # Write storm images for [j]th field/height pair and [i]th
                    # time step.
                    image_file_name_matrix[i, j] = find_storm_image_file(
                        top_directory_name=top_output_dir_name,
                        unix_time_sec=valid_times_unix_sec[i],
                        spc_date_string=unique_spc_date_strings[q],
                        radar_source=radar_source,
                        radar_field_name=radar_field_name_by_pair[j],
                        radar_height_m_asl=radar_height_by_pair_m_asl[j],
                        raise_error_if_missing=False)

                    print 'Writing storm images to "{0:s}"...\n'.format(
                        image_file_name_matrix[i, j])
                    _write_storm_images_only(
                        netcdf_file_name=image_file_name_matrix[i, j],
                        storm_image_matrix=this_storm_image_matrix,
                        storm_ids=these_storm_ids,
                        valid_times_unix_sec=these_times_unix_sec,
                        radar_field_name=radar_field_name_by_pair[j],
                        radar_height_m_asl=radar_height_by_pair_m_asl[j])

                    continue

                if this_date_storm_image_matrix is None:
                    this_date_storm_image_matrix = this_storm_image_matrix + 0.
                else:
                    this_date_storm_image_matrix = numpy.concatenate(
                        (this_date_storm_image_matrix, this_storm_image_matrix),
                        axis=0)

            if one_file_per_time_step:
                continue

            # Write storm images for [j]th field/height pair and [q]th SPC date.
            image_file_name_matrix[q, j] = find_storm_image_file(
                top_directory_name=top_output_dir_name,
                spc_date_string=unique_spc_date_strings[q],
                radar_source=radar_source,
                radar_field_name=radar_field_name_by_pair[j],
                radar_height_m_asl=radar_height_by_pair_m_asl[j],
                raise_error_if_missing=False)

            print 'Writing storm images to "{0:s}"...\n'.format(
                image_file_name_matrix[q, j])
            _write_storm_images_only(
                netcdf_file_name=image_file_name_matrix[q, j],
                storm_image_matrix=this_date_storm_image_matrix,
                storm_ids=this_date_storm_ids,
                valid_times_unix_sec=this_date_valid_times_unix_sec,
                radar_field_name=radar_field_name_by_pair[j],
                radar_height_m_asl=radar_height_by_pair_m_asl[j])

    return image_file_name_matrix


def extract_storm_images_gridrad(
        storm_object_table, top_radar_dir_name, top_output_dir_name,
        one_file_per_time_step=False,
        num_storm_image_rows=DEFAULT_NUM_IMAGE_ROWS,
        num_storm_image_columns=DEFAULT_NUM_IMAGE_COLUMNS,
        radar_field_names=DEFAULT_GRIDRAD_FIELD_NAMES,
        radar_heights_m_asl=DEFAULT_RADAR_HEIGHTS_M_ASL):
    """Extracts storm-centered radar image for each field, height, storm object.

    L = number of storm objects
    F = number of radar fields
    H = number of radar heights
    T = number of time steps with storm objects

    :param storm_object_table: See documentation for
        `extract_storm_images_myrorss_or_mrms`.
    :param top_radar_dir_name: Same.
    :param top_output_dir_name: Same.
    :param one_file_per_time_step: Same.
    :param num_storm_image_rows: Same.
    :param num_storm_image_columns: Same.
    :param radar_field_names: length-F list with names of radar fields.
    :param radar_heights_m_asl: length-H numpy array of radar heights (metres
        above sea level).
    :return: image_file_name_matrix: T-by-F-by-H numpy array of paths to output
        files.
    :raises: ValueError: if grid spacing is not uniform across all files.
    """

    error_checking.assert_is_boolean(one_file_per_time_step)

    _, _ = gridrad_utils.fields_and_refl_heights_to_pairs(
        field_names=radar_field_names, heights_m_asl=radar_heights_m_asl)
    radar_heights_m_asl = numpy.sort(
        numpy.round(radar_heights_m_asl).astype(int))

    valid_times_unix_sec = numpy.unique(
        storm_object_table[tracking_utils.TIME_COLUMN].values)
    valid_time_strings = [time_conversion.unix_sec_to_string(t, TIME_FORMAT)
                          for t in valid_times_unix_sec]

    valid_spc_dates_unix_sec = numpy.array(
        [time_conversion.time_to_spc_date_unix_sec(t)
         for t in valid_times_unix_sec], dtype=int)
    unique_spc_dates_unix_sec = numpy.unique(valid_spc_dates_unix_sec)
    unique_spc_date_strings = [time_conversion.time_to_spc_date_string(t)
                               for t in unique_spc_dates_unix_sec]

    # Find radar files.
    num_times = len(valid_times_unix_sec)
    radar_file_names = [None] * num_times
    for i in range(num_times):
        radar_file_names[i] = gridrad_io.find_file(
            unix_time_sec=valid_times_unix_sec[i],
            top_directory_name=top_radar_dir_name, raise_error_if_missing=True)

    # Initialize output array.
    num_times = len(valid_times_unix_sec)
    num_unique_spc_dates = len(unique_spc_date_strings)
    num_fields = len(radar_field_names)
    num_heights = len(radar_heights_m_asl)

    if one_file_per_time_step:
        image_file_name_matrix = numpy.full(
            (num_times, num_fields, num_heights), '', dtype=object)
    else:
        image_file_name_matrix = numpy.full(
            (num_unique_spc_dates, num_fields, num_heights), '', dtype=object)

    latitude_spacing_deg = None
    longitude_spacing_deg = None

    for q in range(num_unique_spc_dates):
        these_time_indices = numpy.where(
            valid_spc_dates_unix_sec == unique_spc_dates_unix_sec[q])[0]

        for j in range(num_fields):
            for k in range(num_heights):
                this_date_storm_image_matrix = None
                this_date_storm_ids = []
                this_date_valid_times_unix_sec = numpy.array([], dtype=int)

                for i in these_time_indices:

                    # Read metadata for [i]th valid time.
                    this_metadata_dict = (
                        gridrad_io.read_metadata_from_full_grid_file(
                            radar_file_names[i]))

                    this_lat_spacing_deg = rounder.round_to_nearest(
                        this_metadata_dict[radar_utils.LAT_SPACING_COLUMN],
                        GRID_SPACING_TOLERANCE_DEG)
                    this_lng_spacing_deg = rounder.round_to_nearest(
                        this_metadata_dict[radar_utils.LNG_SPACING_COLUMN],
                        GRID_SPACING_TOLERANCE_DEG)

                    if latitude_spacing_deg is None:
                        latitude_spacing_deg = this_lat_spacing_deg + 0.
                        longitude_spacing_deg = this_lng_spacing_deg + 0.

                    if (latitude_spacing_deg != this_lat_spacing_deg or
                            longitude_spacing_deg != this_lng_spacing_deg):
                        error_string = (
                            'First file ("{0:s}") has grid spacing of {1:.4f} '
                            'deg lat, {2:.4f} deg long.  {3:d}th file ("{4:s}")'
                            '  has grid spacing of {5:.4f} deg lat, {6:.4f} deg'
                            ' long.  Grid spacing should be uniform across all '
                            'files.'
                        ).format(radar_file_names[0], latitude_spacing_deg,
                                 longitude_spacing_deg, i, radar_file_names[i],
                                 this_lat_spacing_deg, this_lng_spacing_deg)
                        raise ValueError(error_string)

                    # Find storm objects at [i]th valid time.
                    these_storm_indices = numpy.where(
                        storm_object_table[tracking_utils.TIME_COLUMN].values ==
                        valid_times_unix_sec[i])[0]
                    this_num_storms = len(these_storm_indices)

                    these_storm_ids = storm_object_table[
                        tracking_utils.STORM_ID_COLUMN
                    ].values[these_storm_indices].tolist()
                    these_times_unix_sec = storm_object_table[
                        tracking_utils.TIME_COLUMN
                    ].values[these_storm_indices].astype(int)

                    if not one_file_per_time_step:
                        this_date_storm_ids += these_storm_ids
                        this_date_valid_times_unix_sec = numpy.concatenate((
                            this_date_valid_times_unix_sec,
                            these_times_unix_sec))

                    # Read data for [j]th field at [i]th valid time.
                    print 'Reading "{0:s}" from file: "{1:s}"...'.format(
                        radar_field_names[j], radar_file_names[i])

                    this_field_radar_matrix, these_heights_m_asl, _, _ = (
                        gridrad_io.read_field_from_full_grid_file(
                            radar_file_names[i],
                            field_name=radar_field_names[j],
                            metadata_dict=this_metadata_dict))

                    this_height_index = numpy.where(
                        these_heights_m_asl == radar_heights_m_asl[k])[0]
                    this_field_radar_matrix = this_field_radar_matrix[
                        this_height_index, ...]
                    this_field_radar_matrix[
                        numpy.isnan(this_field_radar_matrix)] = 0.

                    this_storm_image_matrix = numpy.full(
                        (this_num_storms, num_storm_image_rows,
                         num_storm_image_columns), numpy.nan)

                    print (
                        'Extracting storm images for "{0:s}" at {1:d} metres '
                        'ASL and {2:s}...'
                    ).format(
                        radar_field_names[j],
                        numpy.round(int(radar_heights_m_asl[k])),
                        valid_time_strings[i])

                    # Extract storm images for [j]th field at [k]th height and
                    # [i]th time step.
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
                            full_radar_matrix=numpy.flipud(
                                this_field_radar_matrix),
                            center_row=these_center_rows[m],
                            center_column=these_center_columns[m],
                            num_storm_image_rows=num_storm_image_rows,
                            num_storm_image_columns=num_storm_image_columns)

                    if one_file_per_time_step:

                        # Write storm images for [j]th field at [k]th height and
                        # [i]th time step.
                        image_file_name_matrix[i, j, k] = find_storm_image_file(
                            top_directory_name=top_output_dir_name,
                            unix_time_sec=valid_times_unix_sec[i],
                            spc_date_string=unique_spc_date_strings[q],
                            radar_source=radar_utils.GRIDRAD_SOURCE_ID,
                            radar_field_name=radar_field_names[j],
                            radar_height_m_asl=radar_heights_m_asl[k],
                            raise_error_if_missing=False)

                        print 'Writing storm images to "{0:s}"...\n'.format(
                            image_file_name_matrix[i, j, k])
                        _write_storm_images_only(
                            netcdf_file_name=image_file_name_matrix[i, j, k],
                            storm_image_matrix=this_storm_image_matrix,
                            storm_ids=these_storm_ids,
                            valid_times_unix_sec=these_times_unix_sec,
                            radar_field_name=radar_field_names[j],
                            radar_height_m_asl=radar_heights_m_asl[k])

                        continue

                    if this_date_storm_image_matrix is None:
                        this_date_storm_image_matrix = (
                            this_storm_image_matrix + 0.)
                    else:
                        this_date_storm_image_matrix = numpy.concatenate(
                            (this_date_storm_image_matrix,
                             this_storm_image_matrix), axis=0)

                if one_file_per_time_step:
                    continue

                # Write storm images for [j]th field at [k]th height and [q]th
                # SPC date.
                image_file_name_matrix[q, j, k] = find_storm_image_file(
                    top_directory_name=top_output_dir_name,
                    spc_date_string=unique_spc_date_strings[q],
                    radar_source=radar_utils.GRIDRAD_SOURCE_ID,
                    radar_field_name=radar_field_names[j],
                    radar_height_m_asl=radar_heights_m_asl[k],
                    raise_error_if_missing=False)

                print 'Writing storm images to "{0:s}"...\n'.format(
                    image_file_name_matrix[q, j, k])
                _write_storm_images_only(
                    netcdf_file_name=image_file_name_matrix[q, j, k],
                    storm_image_matrix=this_date_storm_image_matrix,
                    storm_ids=this_date_storm_ids,
                    valid_times_unix_sec=this_date_valid_times_unix_sec,
                    radar_field_name=radar_field_names[j],
                    radar_height_m_asl=radar_heights_m_asl[k])

    return image_file_name_matrix


def find_storm_image_file(
        top_directory_name, spc_date_string, radar_source, radar_field_name,
        radar_height_m_asl, unix_time_sec=None, raise_error_if_missing=True):
    """Finds file with storm-centered radar images.

    If `unix_time_sec is None`, this method finds a file with images for one SPC
    date.  Otherwise, finds a file with images for one time step.

    Both file types should be written by `_write_storm_images_only`.

    :param top_directory_name: Name of top-level directory with storm-image
        files.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param radar_source: Data source (must be accepted by
        `radar_utils.check_data_source`).
    :param radar_field_name: Name of radar field (string).
    :param radar_height_m_asl: Height (metres above sea level) of radar field.
    :param unix_time_sec: Valid time.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        raise_error_if_missing = True, this method will error out.
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

    if unix_time_sec is None:
        storm_image_file_name = (
            '{0:s}/{1:s}/{2:s}/{3:s}/{4:05d}_metres_asl/storm_images_{5:s}.nc'
        ).format(
            top_directory_name, radar_source, spc_date_string[:4],
            radar_field_name, numpy.round(int(radar_height_m_asl)),
            spc_date_string)
    else:
        storm_image_file_name = (
            '{0:s}/{1:s}/{2:s}/{3:s}/{4:s}/{5:05d}_metres_asl/'
            'storm_images_{6:s}.nc'
        ).format(
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


def find_storm_label_file(
        storm_image_file_name, raise_error_if_missing=True,
        warn_if_missing=True):
    """Finds file with storm-hazard labels.

    This file should be written by `_write_storm_labels_only`.

    :param storm_image_file_name: Path to storm-image file (should be written by
        `_write_storm_images_only`).  This method will find the corresponding
        label file.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will error out.
    :param warn_if_missing: Boolean flag.  If file is missing,
        `raise_error_if_missing` = False, and `warn_if_missing` = True, will
        print warning message.
    :return: storm_label_file_name: Path to label file.  If file is missing and
        raise_error_if_missing = False, this is the *expected* path.
    :raises: ValueError: if file is missing and raise_error_if_missing = True.
    """

    error_checking.assert_is_string(storm_image_file_name)
    error_checking.assert_is_boolean(raise_error_if_missing)
    if raise_error_if_missing:
        warn_if_missing = False
    else:
        error_checking.assert_is_boolean(warn_if_missing)

    storm_image_dir_name, pathless_image_file_name = os.path.split(
        storm_image_file_name)
    dir_name_parts = storm_image_dir_name.split('/')[:-2]
    storm_label_dir_name = '/'.join(dir_name_parts)

    extensionless_image_file_name, _ = os.path.splitext(
        pathless_image_file_name)
    extensionless_label_file_name = extensionless_image_file_name.replace(
        'images', 'labels')
    storm_label_file_name = '{0:s}/{1:s}.p'.format(
        storm_label_dir_name, extensionless_label_file_name)

    if not os.path.isfile(storm_label_file_name):
        error_string = (
            'PROBLEM.  Cannot find file with storm-hazard labels.  Expected at:'
            ' {0:s}'
        ).format(storm_label_file_name)

        if raise_error_if_missing:
            raise ValueError(error_string)

        if warn_if_missing:
            print error_string

    return storm_label_file_name


def find_many_files_myrorss_or_mrms(
        top_directory_name, radar_source, radar_field_names,
        start_time_unix_sec, end_time_unix_sec, one_file_per_time_step=True,
        reflectivity_heights_m_asl=None, raise_error_if_all_missing=True,
        raise_error_if_any_missing=False):
    """Finds many files containing storm images with MYRORSS or MRMS data.

    If `one_file_per_time_step = True`, this method will look for files with one
    time step each.  Otherwise, will look for files with one SPC date each.

    T = number of time steps or SPC dates (in general, number of values along
        the time dimension)
    C = number of field/height pairs

    :param top_directory_name: Name of top-level directory with storm-image
        files.
    :param radar_source: Source (must be accepted by
        `radar_utils.check_data_source`).
    :param radar_field_names: 1-D list with names of radar fields.
    :param start_time_unix_sec: See documentation for `find_many_files_gridrad`.
    :param end_time_unix_sec: Same.
    :param one_file_per_time_step: Same.
    :param reflectivity_heights_m_asl: 1-D numpy array of heights for field
        "reflectivity_dbz".
    :param raise_error_if_all_missing: Boolean flag.  If no files are found and
        `raise_error_if_all_missing` = True, this method will error out.
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing` = True, this method will error out.
    :return: image_file_name_matrix: T-by-C numpy array of file paths.
    :return: valid_times_unix_sec: length-T numpy array of valid times.  If
        `one_file_per_time_step is False`, valid_times_unix_sec[i] is just a
        time on the [i]th SPC date.
    :raises: ValueError: If no files are found and `raise_error_if_all_missing`
        = True.
    """

    error_checking.assert_is_boolean(one_file_per_time_step)
    error_checking.assert_is_boolean(raise_error_if_all_missing)
    error_checking.assert_is_boolean(raise_error_if_any_missing)

    field_name_by_pair, height_by_pair_m_asl = (
        myrorss_and_mrms_utils.fields_and_refl_heights_to_pairs(
            field_names=radar_field_names, data_source=radar_source,
            refl_heights_m_asl=reflectivity_heights_m_asl))
    num_field_height_pairs = len(field_name_by_pair)

    first_spc_date_string = time_conversion.time_to_spc_date_string(
        start_time_unix_sec)
    last_spc_date_string = time_conversion.time_to_spc_date_string(
        end_time_unix_sec)
    all_spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    if one_file_per_time_step:
        image_file_name_matrix = None
        valid_times_unix_sec = None

        for i in range(len(all_spc_date_strings)):
            print 'Finding storm-image files for SPC date "{0:s}"...'.format(
                all_spc_date_strings[i])

            this_file_name_matrix, these_times_unix_sec = (
                _find_many_files_one_spc_date(
                    top_directory_name=top_directory_name,
                    start_time_unix_sec=start_time_unix_sec,
                    end_time_unix_sec=end_time_unix_sec,
                    spc_date_string=all_spc_date_strings[i],
                    radar_source=radar_source,
                    field_name_by_pair=field_name_by_pair,
                    height_by_pair_m_asl=height_by_pair_m_asl,
                    raise_error_if_all_missing=False,
                    raise_error_if_any_missing=raise_error_if_any_missing))

            if this_file_name_matrix is None:
                continue

            if image_file_name_matrix is None:
                image_file_name_matrix = copy.deepcopy(this_file_name_matrix)
                valid_times_unix_sec = copy.deepcopy(these_times_unix_sec)
            else:
                image_file_name_matrix = numpy.concatenate(
                    (image_file_name_matrix, this_file_name_matrix), axis=0)
                valid_times_unix_sec = numpy.concatenate((
                    valid_times_unix_sec, these_times_unix_sec))

        if raise_error_if_all_missing and image_file_name_matrix is None:
            start_time_string = time_conversion.unix_sec_to_string(
                start_time_unix_sec, TIME_FORMAT)
            end_time_string = time_conversion.unix_sec_to_string(
                end_time_unix_sec, TIME_FORMAT)
            error_string = 'Cannot find any files from {0:s} to {1:s}.'.format(
                start_time_string, end_time_string)
            raise ValueError(error_string)

        return image_file_name_matrix, valid_times_unix_sec

    image_file_name_matrix = None
    valid_spc_date_strings = None
    valid_times_unix_sec = None

    for j in range(num_field_height_pairs):
        print (
            'Finding storm-image files for "{0:s}" at {1:d} metres ASL...'
        ).format(field_name_by_pair[j],
                 int(numpy.round(height_by_pair_m_asl[j])))

        if j == 0:
            image_file_names = []
            valid_spc_date_strings = []

            for i in range(len(all_spc_date_strings)):
                this_file_name = find_storm_image_file(
                    top_directory_name=top_directory_name,
                    spc_date_string=all_spc_date_strings[i],
                    radar_source=radar_source,
                    radar_field_name=field_name_by_pair[j],
                    radar_height_m_asl=height_by_pair_m_asl[j],
                    raise_error_if_missing=raise_error_if_any_missing)

                if not os.path.isfile(this_file_name):
                    continue

                image_file_names.append(this_file_name)
                valid_spc_date_strings.append(all_spc_date_strings[i])

            num_times = len(image_file_names)
            if num_times == 0:
                if raise_error_if_all_missing:
                    error_string = (
                        'Cannot find any files from SPC dates "{0:s}" to '
                        '"{1:s}".'
                    ).format(all_spc_date_strings[0],
                             all_spc_date_strings[-1])
                    raise ValueError(error_string)

                return None, None

            image_file_name_matrix = numpy.full(
                (num_times, num_field_height_pairs), '', dtype=object)
            image_file_name_matrix[:, j] = numpy.array(
                image_file_names, dtype=object)

            valid_times_unix_sec = numpy.array(
                [time_conversion.spc_date_string_to_unix_sec(s)
                 for s in valid_spc_date_strings], dtype=int)

        else:
            for i in range(len(valid_spc_date_strings)):
                image_file_name_matrix[i, j] = find_storm_image_file(
                    top_directory_name=top_directory_name,
                    spc_date_string=valid_spc_date_strings[i],
                    radar_source=radar_source,
                    radar_field_name=field_name_by_pair[j],
                    radar_height_m_asl=height_by_pair_m_asl[j],
                    raise_error_if_missing=raise_error_if_any_missing)

                if not os.path.isfile(image_file_name_matrix[i, j]):
                    image_file_name_matrix[i, j] = ''

    return image_file_name_matrix, valid_times_unix_sec


def find_many_files_gridrad(
        top_directory_name, radar_field_names, radar_heights_m_asl,
        start_time_unix_sec, end_time_unix_sec, one_file_per_time_step=True,
        raise_error_if_all_missing=True):
    """Finds many files containing storm images with GridRad data.

    If `one_file_per_time_step = True`, this method will look for files with one
    time step each.  Otherwise, will look for files with one SPC date each.

    T = number of time steps or SPC dates (in general, number of values along
        the time dimension)
    F = number of radar fields
    H = number of radar heights

    :param top_directory_name: Name of top-level directory with storm-image
        files.
    :param radar_field_names: length-F list with names of radar fields.
    :param radar_heights_m_asl: length-H numpy array of radar heights (metres
        above sea level).
    :param start_time_unix_sec: Start time.  This method will find all files
        from `start_time_unix_sec`...`end_time_unix_sec`.  If
        `one_file_per_time_step is False`, this can be any time on the first SPC
        date.
    :param end_time_unix_sec: End time.  If `one_file_per_time_step is False`,
        this can be any time on the last SPC date.
    :param one_file_per_time_step: See general discussion above.
    :param raise_error_if_all_missing: Boolean flag.  If no files are found and
        `raise_error_if_all_missing` = True, this method will error out.
    :return: image_file_name_matrix: T-by-F-by-H numpy array of file paths.
    :return: valid_times_unix_sec: length-T numpy array of valid times.  If
        `one_file_per_time_step is False`, valid_times_unix_sec[i] is just a
        time on the [i]th SPC date.
    :raises: ValueError: If no files are found and `raise_error_if_all_missing`
        = True.
    """

    _, _ = gridrad_utils.fields_and_refl_heights_to_pairs(
        field_names=radar_field_names, heights_m_asl=radar_heights_m_asl)

    error_checking.assert_is_boolean(one_file_per_time_step)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

    if one_file_per_time_step:
        all_times_unix_sec = time_periods.range_and_interval_to_list(
            start_time_unix_sec=start_time_unix_sec,
            end_time_unix_sec=end_time_unix_sec,
            time_interval_sec=GRIDRAD_TIME_INTERVAL_SEC, include_endpoint=True)

        all_spc_date_strings = [time_conversion.time_to_spc_date_string(t)
                                for t in all_times_unix_sec]

    else:
        first_spc_date_string = time_conversion.time_to_spc_date_string(
            start_time_unix_sec)
        last_spc_date_string = time_conversion.time_to_spc_date_string(
            end_time_unix_sec)

        all_spc_date_strings = time_conversion.get_spc_dates_in_range(
            first_spc_date_string=first_spc_date_string,
            last_spc_date_string=last_spc_date_string)
        all_times_unix_sec = numpy.array(
            [time_conversion.spc_date_string_to_unix_sec(s)
             for s in all_spc_date_strings], dtype=int)

    image_file_name_matrix = None
    valid_times_unix_sec = None
    valid_spc_date_strings = None
    num_fields = len(radar_field_names)
    num_heights = len(radar_heights_m_asl)

    for j in range(num_fields):
        for k in range(num_heights):
            print (
                'Finding storm-image files for "{0:s}" at {1:d} metres ASL...'
            ).format(radar_field_names[j],
                     int(numpy.round(radar_heights_m_asl[k])))

            if j == 0 and k == 0:
                image_file_names = []
                valid_times_unix_sec = []
                valid_spc_date_strings = []

                for i in range(len(all_times_unix_sec)):
                    if one_file_per_time_step:
                        this_time_unix_sec = all_times_unix_sec[i]
                    else:
                        this_time_unix_sec = None

                    this_file_name = find_storm_image_file(
                        top_directory_name=top_directory_name,
                        unix_time_sec=this_time_unix_sec,
                        spc_date_string=all_spc_date_strings[i],
                        radar_source=radar_utils.GRIDRAD_SOURCE_ID,
                        radar_field_name=radar_field_names[j],
                        radar_height_m_asl=radar_heights_m_asl[k],
                        raise_error_if_missing=False)

                    if not os.path.isfile(this_file_name):
                        continue

                    image_file_names.append(this_file_name)
                    valid_times_unix_sec.append(all_times_unix_sec[i])
                    valid_spc_date_strings.append(all_spc_date_strings[i])

                num_times = len(image_file_names)
                if num_times == 0:
                    if raise_error_if_all_missing and one_file_per_time_step:
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

                    if raise_error_if_all_missing:
                        error_string = (
                            'Cannot find any files from SPC dates "{0:s}" to '
                            '"{1:s}".'
                        ).format(all_spc_date_strings[0],
                                 all_spc_date_strings[-1])
                        raise ValueError(error_string)

                    return None, None

                image_file_name_matrix = numpy.full(
                    (num_times, num_fields, num_heights), '', dtype=object)
                image_file_name_matrix[:, j, k] = numpy.array(
                    image_file_names, dtype=object)
                valid_times_unix_sec = numpy.array(
                    valid_times_unix_sec, dtype=int)

            else:
                for i in range(len(valid_times_unix_sec)):
                    if one_file_per_time_step:
                        this_time_unix_sec = valid_times_unix_sec[i]
                    else:
                        this_time_unix_sec = None

                    image_file_name_matrix[i, j, k] = find_storm_image_file(
                        top_directory_name=top_directory_name,
                        unix_time_sec=this_time_unix_sec,
                        spc_date_string=valid_spc_date_strings[i],
                        radar_source=radar_utils.GRIDRAD_SOURCE_ID,
                        radar_field_name=radar_field_names[j],
                        radar_height_m_asl=radar_heights_m_asl[k],
                        raise_error_if_missing=True)

    return image_file_name_matrix, valid_times_unix_sec


def write_storm_images_and_labels(
        image_file_name, label_file_name, storm_image_matrix, storm_ids,
        valid_times_unix_sec, radar_field_name, radar_height_m_asl,
        storm_to_winds_table=None, storm_to_tornadoes_table=None):
    """Writes storm-centered radar images and hazard labels to files.

    Images should be created by `extract_storm_image`.
    Labels should be created by `extract_storm_labels`.

    :param image_file_name: Path to NetCDF output file.
    :param label_file_name: Path to Pickle output file.
    :param storm_image_matrix: See documentation for `_check_storm_images`.
    :param storm_ids: Same.
    :param valid_times_unix_sec: Same.
    :param radar_field_name: Same.
    :param radar_height_m_asl: Same.
    :param storm_to_winds_table: See doc for `_check_storm_labels`.
    :param storm_to_tornadoes_table: Same.
    """

    _check_storm_images(
        storm_image_matrix=storm_image_matrix, storm_ids=storm_ids,
        valid_times_unix_sec=valid_times_unix_sec,
        radar_field_name=radar_field_name,
        radar_height_m_asl=radar_height_m_asl)

    _check_storm_labels(
        storm_ids=storm_ids, valid_times_unix_sec=valid_times_unix_sec,
        storm_to_winds_table=storm_to_winds_table,
        storm_to_tornadoes_table=storm_to_tornadoes_table)

    _write_storm_images_only(
        netcdf_file_name=image_file_name, storm_image_matrix=storm_image_matrix,
        storm_ids=storm_ids, valid_times_unix_sec=valid_times_unix_sec,
        radar_field_name=radar_field_name,
        radar_height_m_asl=radar_height_m_asl)

    _write_storm_labels_only(
        pickle_file_name=label_file_name,
        storm_to_winds_table=storm_to_winds_table,
        storm_to_tornadoes_table=storm_to_tornadoes_table)


def read_storm_images_only(
        netcdf_file_name, return_images=True, storm_ids_to_keep=None,
        valid_times_to_keep_unix_sec=None):
    """Reads storm-centered radar images from NetCDF file.

    p = number of storm objects to keep

    If `storm_ids_to_keep is None` or `valid_times_to_keep_unix_sec is None`,
    will keep all storm objects.

    :param netcdf_file_name: Path to input file.
    :param return_images: Boolean flag.  If True, will return storm images +
        metadata.  If False, will return only metadata.
    :param storm_ids_to_keep: length-p list with string IDs of storm objects to
        keep.
    :param valid_times_to_keep_unix_sec: length-p numpy array with valid times
        of storm objects to keep.
    :return: storm_image_dict: Dictionary with the following keys.
    storm_image_dict['storm_image_matrix']: See documentation for
        `_check_storm_images`.
    storm_image_dict['storm_ids']: Same.
    storm_image_dict['valid_times_unix_sec']: Same.
    storm_image_dict['radar_field_name']: Same.
    storm_image_dict['radar_height_m_asl']: Same.
    """

    error_checking.assert_is_boolean(return_images)
    netcdf_dataset = netcdf_io.open_netcdf(
        netcdf_file_name=netcdf_file_name, raise_error_if_fails=True)

    radar_field_name = str(getattr(netcdf_dataset, RADAR_FIELD_NAME_KEY))
    radar_height_m_asl = getattr(netcdf_dataset, RADAR_HEIGHT_KEY)
    storm_ids = netCDF4.chartostring(netcdf_dataset.variables[STORM_IDS_KEY][:])
    storm_ids = [str(s) for s in storm_ids]

    try:
        valid_times_unix_sec = numpy.array(
            netcdf_dataset.variables[VALID_TIMES_KEY][:], dtype=int)
    except:
        valid_times_unix_sec = numpy.full(
            len(storm_ids), getattr(netcdf_dataset, 'unix_time_sec'), dtype=int)

    if not return_images:
        return {
            STORM_IDS_KEY: storm_ids,
            VALID_TIMES_KEY: valid_times_unix_sec,
            RADAR_FIELD_NAME_KEY: radar_field_name,
            RADAR_HEIGHT_KEY: radar_height_m_asl
        }

    num_storm_objects = len(storm_ids)

    if storm_ids_to_keep is None or valid_times_to_keep_unix_sec is None:
        indices_to_keep = numpy.linspace(
            0, num_storm_objects - 1, num=num_storm_objects, dtype=int)
    else:
        error_checking.assert_is_string_list(storm_ids_to_keep)
        error_checking.assert_is_numpy_array(
            numpy.array(storm_ids_to_keep), num_dimensions=1)
        num_storm_objects_to_keep = len(storm_ids_to_keep)

        error_checking.assert_is_integer_numpy_array(
            valid_times_to_keep_unix_sec)
        error_checking.assert_is_numpy_array(
            valid_times_to_keep_unix_sec,
            exact_dimensions=numpy.array([num_storm_objects_to_keep]))

        indices_to_keep = _find_storm_objects(
            all_storm_ids=storm_ids,
            all_valid_times_unix_sec=valid_times_unix_sec,
            storm_ids_to_keep=storm_ids_to_keep,
            valid_times_to_keep_unix_sec=valid_times_to_keep_unix_sec)

        storm_ids = [storm_ids[i] for i in indices_to_keep]
        valid_times_unix_sec = valid_times_unix_sec[indices_to_keep]

    storm_image_matrix = numpy.array(
        netcdf_dataset.variables[STORM_IMAGE_MATRIX_KEY][
            indices_to_keep, ...])
    netcdf_dataset.close()

    _check_storm_images(
        storm_image_matrix=storm_image_matrix, storm_ids=storm_ids,
        valid_times_unix_sec=valid_times_unix_sec,
        radar_field_name=radar_field_name,
        radar_height_m_asl=radar_height_m_asl)

    return {
        STORM_IMAGE_MATRIX_KEY: storm_image_matrix,
        STORM_IDS_KEY: storm_ids,
        VALID_TIMES_KEY: valid_times_unix_sec,
        RADAR_FIELD_NAME_KEY: radar_field_name,
        RADAR_HEIGHT_KEY: radar_height_m_asl
    }


def read_storm_images_and_labels(
        image_file_name, label_file_name, return_label_name=None,
        num_storm_objects_class_dict=None):
    """Reads storm-centered radar images and hazard labels from files.

    Both files should be written by `write_storm_images_and_labels`.

    If `num_storm_objects_class_dict is not None` and no desired storm objects
    are found, this method will return None.

    :param image_file_name: Path to NetCDF input file.
    :param label_file_name: Path to Pickle input file.
    :param return_label_name: Name of label (target variable) to return for each
        storm object.  Must be a column in `storm_to_winds_table` or
        `storm_to_tornadoes_table`.
    :param num_storm_objects_class_dict:
        [used only if `return_label_name is not None`]
        Dictionary, where each key is a class integer (-2 for dead storms) and
        each value is the corresponding number of storm objects to return.

    :return storm_image_dict: Dictionary with the following keys.
    storm_image_dict['storm_image_matrix']: See documentation for
        `_check_storm_images`.
    storm_image_dict['storm_ids']: Same.
    storm_image_dict['valid_times_unix_sec']: Same.
    storm_image_dict['radar_field_name']: Same.
    storm_image_dict['radar_height_m_asl']: Same.

    If `return_label_name is None`, the following keys are included.

    storm_image_dict['storm_to_winds_table']: See doc for `_check_storm_labels`.
    storm_image_dict['storm_to_tornadoes_table']: Same.

    If `return_label_name is not None`, the following keys are included.

    storm_image_dict['label_values']: length-L numpy array with label (target)
        for each storm object, where L = number of storm objects.
    """

    storm_to_winds_table, storm_to_tornadoes_table = (
        _read_storm_labels_only(label_file_name))

    if return_label_name is None:
        storm_image_dict = read_storm_images_only(
            netcdf_file_name=image_file_name, return_images=True)

        _check_storm_labels(
            storm_ids=storm_image_dict[STORM_IDS_KEY],
            valid_times_unix_sec=storm_image_dict[VALID_TIMES_KEY],
            storm_to_winds_table=storm_to_winds_table,
            storm_to_tornadoes_table=storm_to_tornadoes_table)

        storm_image_dict.update({
            STORM_TO_WINDS_TABLE_KEY: storm_to_winds_table,
            STORM_TO_TORNADOES_TABLE_KEY: storm_to_tornadoes_table
        })
        return storm_image_dict

    storm_image_dict = read_storm_images_only(
        netcdf_file_name=image_file_name, return_images=False)
    label_values = extract_storm_labels_with_name(
        storm_ids=storm_image_dict[STORM_IDS_KEY],
        valid_times_unix_sec=storm_image_dict[VALID_TIMES_KEY],
        label_name=return_label_name, storm_to_winds_table=storm_to_winds_table,
        storm_to_tornadoes_table=storm_to_tornadoes_table)

    if num_storm_objects_class_dict is None:
        storm_image_dict = read_storm_images_only(
            netcdf_file_name=image_file_name, return_images=True)
    else:
        indices_to_keep = _filter_storm_objects_by_label(
            label_values=label_values,
            num_storm_objects_class_dict=num_storm_objects_class_dict)
        if not len(indices_to_keep):
            return None

        storm_ids_to_keep = [
            storm_image_dict[STORM_IDS_KEY][i] for i in indices_to_keep]
        valid_times_to_keep_unix_sec = storm_image_dict[
            VALID_TIMES_KEY][indices_to_keep]

        storm_image_dict = read_storm_images_only(
            netcdf_file_name=image_file_name, return_images=True,
            storm_ids_to_keep=storm_ids_to_keep,
            valid_times_to_keep_unix_sec=valid_times_to_keep_unix_sec)
        label_values = label_values[indices_to_keep]

    storm_image_dict.update({LABEL_VALUES_KEY: label_values})
    return storm_image_dict


def extract_storm_labels(
        storm_ids, valid_times_unix_sec, storm_to_winds_table=None,
        storm_to_tornadoes_table=None):
    """Extracts all labels (target variables) for each storm object.

    :param storm_ids: See documentation for `_check_storm_labels`.
    :param valid_times_unix_sec: Same.
    :param storm_to_winds_table: Same.
    :param storm_to_tornadoes_table: Same.
    :return: relevant_storm_to_winds_table: Same.
    :return: relevant_storm_to_tornadoes_table: Same.
    :raises: ValueError: if `storm_to_winds_table` and
        `storm_to_tornadoes_table` are both None.
    """

    if storm_to_winds_table is None and storm_to_tornadoes_table is None:
        raise ValueError('`storm_to_winds_table` and `storm_to_tornadoes_table`'
                         ' cannot both be None.')

    return _check_storm_labels(
        storm_ids=storm_ids, valid_times_unix_sec=valid_times_unix_sec,
        storm_to_winds_table=storm_to_winds_table,
        storm_to_tornadoes_table=storm_to_tornadoes_table)


def extract_storm_labels_with_name(
        storm_ids, valid_times_unix_sec, label_name, storm_to_winds_table=None,
        storm_to_tornadoes_table=None):
    """Extracts one label (target variable) for each storm object.

    L = number of storm objects

    :param storm_ids: See documentation for `_check_storm_labels`.
    :param valid_times_unix_sec: Same.
    :param label_name: Name of target variable.
    :param storm_to_winds_table: See doc for `_check_storm_labels`.
    :param storm_to_tornadoes_table: Same.
    :return: label_values: length-L numpy array of integers.  label_values[i] is
        the class for the [i]th storm object.
    """

    metadata_dict = labels.column_name_to_label_params(label_name)
    event_type_string = metadata_dict[labels.EVENT_TYPE_KEY]

    if event_type_string == events2storms.TORNADO_EVENT_TYPE_STRING:
        _, relevant_storm_to_tornadoes_table = _check_storm_labels(
            storm_ids=storm_ids, valid_times_unix_sec=valid_times_unix_sec,
            storm_to_winds_table=None,
            storm_to_tornadoes_table=storm_to_tornadoes_table)

        return relevant_storm_to_tornadoes_table[label_name].values

    relevant_storm_to_winds_table, _ = _check_storm_labels(
        storm_ids=storm_ids, valid_times_unix_sec=valid_times_unix_sec,
        storm_to_winds_table=storm_to_winds_table,
        storm_to_tornadoes_table=None)
    return relevant_storm_to_winds_table[label_name].values


def attach_labels_to_storm_images(
        image_file_name_matrix, storm_to_winds_table=None,
        storm_to_tornadoes_table=None):
    """Attaches labels (target variables) to each storm-centered radar image.

    T = number of time steps
    C = number of radar variables (field/height pairs)

    :param image_file_name_matrix: T-by-C numpy array of paths to storm-image
        files (should be written by `_write_storm_images_only`).
    :param storm_to_winds_table: See documentation for `extract_storm_labels`.
    :param storm_to_tornadoes_table: Same.
    """

    error_checking.assert_is_numpy_array(
        image_file_name_matrix, num_dimensions=2)
    num_times = image_file_name_matrix.shape[0]
    num_field_height_pairs = image_file_name_matrix.shape[1]

    for i in range(num_times):
        for j in range(num_field_height_pairs):
            if image_file_name_matrix[i, j] == '':
                continue

            print 'Reading storm-centered radar images from: "{0:s}"...'.format(
                image_file_name_matrix[i, j])
            this_storm_image_dict = read_storm_images_only(
                netcdf_file_name=image_file_name_matrix[i, j])

            print 'Attaching labels to storm images...'
            this_storm_to_winds_table, this_storm_to_tornadoes_table = (
                extract_storm_labels(
                    storm_ids=this_storm_image_dict[STORM_IDS_KEY],
                    valid_times_unix_sec=this_storm_image_dict[VALID_TIMES_KEY],
                    storm_to_winds_table=storm_to_winds_table,
                    storm_to_tornadoes_table=storm_to_tornadoes_table))

            this_label_file_name = find_storm_label_file(
                storm_image_file_name=image_file_name_matrix[i, j],
                raise_error_if_missing=False)

            print 'Writing storm-hazard labels to: "{0:s}"...\n'.format(
                this_label_file_name)
            _write_storm_labels_only(
                pickle_file_name=this_label_file_name,
                storm_to_winds_table=this_storm_to_winds_table,
                storm_to_tornadoes_table=this_storm_to_tornadoes_table)

            break
