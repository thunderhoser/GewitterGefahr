"""Methods for extracting radar subgrids.

These are usually centered around a storm object.
"""

import pickle
import numpy
from gewittergefahr.gg_io import myrorss_and_mrms_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M%S'

MIN_ROW_IN_SUBGRID_COLUMN = 'min_row_in_subgrid'
MAX_ROW_IN_SUBGRID_COLUMN = 'max_row_in_subgrid'
MIN_COLUMN_IN_SUBGRID_COLUMN = 'min_column_in_subgrid'
MAX_COLUMN_IN_SUBGRID_COLUMN = 'max_column_in_subgrid'
NUM_PADDED_ROWS_AT_START_COLUMN = 'num_padded_rows_at_start'
NUM_PADDED_ROWS_AT_END_COLUMN = 'num_padded_rows_at_end'
NUM_PADDED_COLUMNS_AT_START_COLUMN = 'num_padded_columns_at_start'
NUM_PADDED_COLUMNS_AT_END_COLUMN = 'num_padded_columns_at_end'

DEFAULT_NUM_ROWS_PER_IMAGE = 32
DEFAULT_NUM_COLUMNS_PER_IMAGE = 32
MIN_ROWS_PER_IMAGE = 8
MIN_COLUMNS_PER_IMAGE = 8

DEFAULT_RADAR_FIELD_NAMES = [
    radar_utils.ECHO_TOP_18DBZ_NAME, radar_utils.ECHO_TOP_50DBZ_NAME,
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME,
    radar_utils.REFL_COLUMN_MAX_NAME, radar_utils.MESH_NAME,
    radar_utils.REFL_0CELSIUS_NAME, radar_utils.REFL_M10CELSIUS_NAME,
    radar_utils.REFL_M20CELSIUS_NAME, radar_utils.REFL_LOWEST_ALTITUDE_NAME,
    radar_utils.SHI_NAME, radar_utils.VIL_NAME]

AZIMUTHAL_SHEAR_FIELD_NAMES = [
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME]


def _center_points_latlng_to_rowcol(center_latitudes_deg, center_longitudes_deg,
                                    nw_grid_point_lat_deg=None,
                                    nw_grid_point_lng_deg=None,
                                    lat_spacing_deg=None, lng_spacing_deg=None):
    """Converts center points from lat-long to row-column coordinates.

    Each "center point" is meant for input to extract_points_as_2d_array.

    P = number of center points

    :param center_latitudes_deg: length-P numpy array of latitudes (deg N).
    :param center_longitudes_deg: length-P numpy array of longitudes (deg E).
    :param nw_grid_point_lat_deg: Latitude (deg N) of northwesternmost grid
        point.
    :param nw_grid_point_lng_deg: Longitude (deg E) of northwesternmost grid
        point.
    :param lat_spacing_deg: Spacing (deg N) between adjacent rows.
    :param lng_spacing_deg: Spacing (deg E) between adjacent columns.
    :return: center_row_indices: Row indices (half-integers) of center points.
    :return: center_column_indices: Column indices (half-integers) of center
        points.
    """

    center_row_indices, center_column_indices = radar_utils.latlng_to_rowcol(
        center_latitudes_deg, center_longitudes_deg,
        nw_grid_point_lat_deg=nw_grid_point_lat_deg,
        nw_grid_point_lng_deg=nw_grid_point_lng_deg,
        lat_spacing_deg=lat_spacing_deg, lng_spacing_deg=lng_spacing_deg)

    return (rounder.round_to_half_integer(center_row_indices),
            rounder.round_to_half_integer(center_column_indices))


def _get_rowcol_indices_for_subgrid(num_rows_in_full_grid=None,
                                    num_columns_in_full_grid=None,
                                    center_row_index=None,
                                    center_column_index=None,
                                    num_rows_in_subgrid=None,
                                    num_columns_in_subgrid=None):
    """Generates row-column indices for subgrid.

    These row-column indices are meant for input to extract_points_as_2d_array.

    :param num_rows_in_full_grid: Number of rows in full grid (integer).
    :param num_columns_in_full_grid: Number of columns in full grid (integer).
    :param center_row_index: Row index (half-integer) at center point of
        subgrid.
    :param center_column_index: Column index (half-integer) at center point of
        subgrid.
    :param num_rows_in_subgrid: Number of rows in subgrid (even integer).
    :param num_columns_in_subgrid: Number of columns in subgrid (even integer).

    :return: subgrid_dict: Dictionary with the following keys.
    subgrid_dict['min_row_in_subgrid']: Minimum row (integer) in subgrid.  If
        min_row_in_subgrid = i, this means the [i]th row of the full grid is the
        first row in the subgrid.
    subgrid_dict['max_row_in_subgrid']: Maximum row (integer) in subgrid.
    subgrid_dict['min_column_in_subgrid']: Minimum column (integer) in subgrid.
    subgrid_dict['max_column_in_subgrid']: Maximum column (integer) in subgrid.
    subgrid_dict['num_padded_rows_at_start']: Number of NaN rows at beginning
        (top) of subgrid.
    subgrid_dict['num_padded_rows_at_end']: Number of NaN rows at end (bottom)
        of subgrid.
    subgrid_dict['num_padded_columns_at_start']: Number of NaN columns at
        beginning (left) of subgrid.
    subgrid_dict['num_padded_columns_at_end']: Number of NaN columns at end
        (right) of subgrid.
    """

    min_row_in_subgrid = int(numpy.ceil(
        center_row_index - num_rows_in_subgrid / 2))
    if min_row_in_subgrid >= 0:
        num_padded_rows_at_start = 0
    else:
        num_padded_rows_at_start = -1 * min_row_in_subgrid
        min_row_in_subgrid = 0

    max_row_in_subgrid = int(numpy.floor(
        center_row_index + num_rows_in_subgrid / 2))
    if max_row_in_subgrid <= num_rows_in_full_grid - 1:
        num_padded_rows_at_end = 0
    else:
        num_padded_rows_at_end = max_row_in_subgrid - (
            num_rows_in_full_grid - 1)
        max_row_in_subgrid = num_rows_in_full_grid - 1

    min_column_in_subgrid = int(numpy.ceil(
        center_column_index - num_columns_in_subgrid / 2))
    if min_column_in_subgrid >= 0:
        num_padded_columns_at_start = 0
    else:
        num_padded_columns_at_start = -1 * min_column_in_subgrid
        min_column_in_subgrid = 0

    max_column_in_subgrid = int(numpy.floor(
        center_column_index + num_columns_in_subgrid / 2))
    if max_column_in_subgrid <= num_columns_in_full_grid - 1:
        num_padded_columns_at_end = 0
    else:
        num_padded_columns_at_end = max_column_in_subgrid - (
            num_columns_in_full_grid - 1)
        max_column_in_subgrid = num_columns_in_full_grid - 1

    return {
        MIN_ROW_IN_SUBGRID_COLUMN: min_row_in_subgrid,
        MAX_ROW_IN_SUBGRID_COLUMN: max_row_in_subgrid,
        MIN_COLUMN_IN_SUBGRID_COLUMN: min_column_in_subgrid,
        MAX_COLUMN_IN_SUBGRID_COLUMN: max_column_in_subgrid,
        NUM_PADDED_ROWS_AT_START_COLUMN: num_padded_rows_at_start,
        NUM_PADDED_ROWS_AT_END_COLUMN: num_padded_rows_at_end,
        NUM_PADDED_COLUMNS_AT_START_COLUMN: num_padded_columns_at_start,
        NUM_PADDED_COLUMNS_AT_END_COLUMN: num_padded_columns_at_end}


def _check_storm_images(
        image_matrix, storm_ids, radar_field_name_by_pair,
        radar_height_by_pair_m_asl):
    """Checks storm-centered radar images for errors.

    These images should be created by `get_images_for_storm_objects`.

    S = number of storm objects
    F = number of radar fields (variable/height pairs)
    M = number of rows in each image
    N = number of columns in each image

    :param image_matrix: 4-D numpy array (F x S x M x N) with image for each
        radar field and storm object.
    :param storm_ids: length-S list of storm IDs (strings).
    :param radar_field_name_by_pair: length-F list with names of radar fields.
    :param radar_height_by_pair_m_asl: length-F list with heights (metres above
        sea level) of radar fields.
    """

    error_checking.assert_is_string_list(storm_ids)
    error_checking.assert_is_numpy_array(
        numpy.array(storm_ids), num_dimensions=1)
    num_storm_objects = len(storm_ids)

    error_checking.assert_is_string_list(radar_field_name_by_pair)
    error_checking.assert_is_numpy_array(
        numpy.array(radar_field_name_by_pair), num_dimensions=1)
    num_radar_fields = len(radar_field_name_by_pair)

    error_checking.assert_is_numpy_array(
        radar_height_by_pair_m_asl,
        exact_dimensions=numpy.array([num_radar_fields]))
    error_checking.assert_is_geq_numpy_array(radar_height_by_pair_m_asl, 0)

    error_checking.assert_is_numpy_array(image_matrix, num_dimensions=4)
    error_checking.assert_is_numpy_array(
        image_matrix, exact_dimensions=numpy.array(
            [num_radar_fields, num_storm_objects, image_matrix.shape[2],
             image_matrix.shape[3]]))


def extract_radar_subgrid(field_matrix, center_row_index=None,
                          center_column_index=None, num_rows_in_subgrid=None,
                          num_columns_in_subgrid=None):
    """Extracts contiguous subset of radar field.

    M = number of rows (unique grid-point latitudes) in full grid
    N = number of columns (unique grid-point longitudes) in full grid
    m = number of rows in subgrid
    n = number of columns in subgrid

    :param field_matrix: M-by-N numpy array with values of a single radar field.
    :param center_row_index: Row index (half-integer) at center of subgrid.
    :param center_column_index: Column index (half-integer) at center of
        subgrid.
    :param num_rows_in_subgrid: Number of rows in subgrid.
    :param num_columns_in_subgrid: Number of columns in subgrid.
    :return: field_submatrix: m-by-n numpy array with values in subgrid.
    """

    error_checking.assert_is_real_numpy_array(field_matrix)
    error_checking.assert_is_numpy_array(field_matrix, num_dimensions=2)
    num_rows_in_full_grid = field_matrix.shape[0]
    num_columns_in_full_grid = field_matrix.shape[1]

    subgrid_dict = _get_rowcol_indices_for_subgrid(
        num_rows_in_full_grid=num_rows_in_full_grid,
        num_columns_in_full_grid=num_columns_in_full_grid,
        center_row_index=center_row_index,
        center_column_index=center_column_index,
        num_rows_in_subgrid=num_rows_in_subgrid,
        num_columns_in_subgrid=num_columns_in_subgrid)

    min_row_in_subgrid = subgrid_dict[MIN_ROW_IN_SUBGRID_COLUMN]
    max_row_in_subgrid = subgrid_dict[MAX_ROW_IN_SUBGRID_COLUMN]
    min_column_in_subgrid = subgrid_dict[MIN_COLUMN_IN_SUBGRID_COLUMN]
    max_column_in_subgrid = subgrid_dict[MAX_COLUMN_IN_SUBGRID_COLUMN]

    field_submatrix = (
        field_matrix[
            min_row_in_subgrid:(max_row_in_subgrid + 1),
            min_column_in_subgrid:(max_column_in_subgrid + 1)])

    num_padded_rows_at_start = subgrid_dict[NUM_PADDED_ROWS_AT_START_COLUMN]
    num_padded_rows_at_end = subgrid_dict[NUM_PADDED_ROWS_AT_END_COLUMN]
    num_padded_columns_at_start = subgrid_dict[
        NUM_PADDED_COLUMNS_AT_START_COLUMN]
    num_padded_columns_at_end = subgrid_dict[NUM_PADDED_COLUMNS_AT_END_COLUMN]

    if num_padded_rows_at_start > 0:
        nan_matrix = numpy.full(
            (num_padded_rows_at_start, field_submatrix.shape[1]), numpy.nan)
        field_submatrix = numpy.vstack((nan_matrix, field_submatrix))

    if num_padded_rows_at_end > 0:
        nan_matrix = numpy.full(
            (num_padded_rows_at_end, field_submatrix.shape[1]), numpy.nan)
        field_submatrix = numpy.vstack((field_submatrix, nan_matrix))

    if num_padded_columns_at_start > 0:
        nan_matrix = numpy.full(
            (field_submatrix.shape[0], num_padded_columns_at_start), numpy.nan)
        field_submatrix = numpy.hstack((nan_matrix, field_submatrix))

    if num_padded_columns_at_end > 0:
        nan_matrix = numpy.full(
            (field_submatrix.shape[0], num_padded_columns_at_end), numpy.nan)
        field_submatrix = numpy.hstack((field_submatrix, nan_matrix))

    return field_submatrix


def get_images_for_storm_objects(
        storm_object_table, top_radar_directory_name,
        num_rows_per_image=DEFAULT_NUM_ROWS_PER_IMAGE,
        num_columns_per_image=DEFAULT_NUM_COLUMNS_PER_IMAGE,
        radar_field_names=DEFAULT_RADAR_FIELD_NAMES,
        reflectivity_heights_m_asl=None,
        radar_data_source=radar_utils.MYRORSS_SOURCE_ID):
    """Extracts image (subgrid) for each radar field and each storm object.

    Images for storm object s will have the same center as s.

    N = number of storm objects
    F = number of radar fields
    T = number of time steps with storm objects

    :param storm_object_table: N-row pandas DataFrame with the following
        columns.  Each row is one storm object.
    :param top_radar_directory_name: Name of top-level directory with radar
        data.
    :param num_rows_per_image: Number of rows in each image (subgrid).  We
        recommend that you make this a power of 2 (examples: 8, 16, 32, 64,
        etc.).
    :param num_columns_per_image: Number of columns in each image (subgrid).  We
        recommend that you make this a power of 2 (examples: 8, 16, 32, 64,
        etc.).
    :param radar_field_names: length-F list with names of radar fields.
    :param reflectivity_heights_m_asl: 1-D numpy array of heights (metres above
        sea level) for radar field "reflectivity_dbz".  If "reflectivity_dbz" is
        not one of the `radar_field_names`, leave this argument as None.
    :param radar_data_source: Source of radar data (examples: "myrorss",
        "mrms").
    :return: image_file_names: length-T list of paths to output files (one for
        each time step with storm objects).
    """

    error_checking.assert_is_integer(num_rows_per_image)
    error_checking.assert_is_geq(num_rows_per_image, MIN_ROWS_PER_IMAGE)
    error_checking.assert_is_integer(num_columns_per_image)
    error_checking.assert_is_geq(num_columns_per_image, MIN_COLUMNS_PER_IMAGE)

    file_dictionary = myrorss_and_mrms_io.find_many_raw_files(
        valid_times_unix_sec=
        storm_object_table[tracking_utils.TIME_COLUMN].values,
        spc_dates_unix_sec=
        storm_object_table[tracking_utils.SPC_DATE_COLUMN].values,
        data_source=radar_data_source, field_names=radar_field_names,
        top_directory_name=top_radar_directory_name,
        reflectivity_heights_m_asl=reflectivity_heights_m_asl)

    radar_file_names_2d_list = file_dictionary[
        myrorss_and_mrms_io.RADAR_FILE_NAME_LIST_KEY]
    unique_storm_times_unix_sec = file_dictionary[
        myrorss_and_mrms_io.UNIQUE_TIMES_KEY]
    unique_spc_dates_unix_sec = file_dictionary[
        myrorss_and_mrms_io.UNIQUE_SPC_DATES_KEY]
    radar_field_name_by_pair = file_dictionary[
        myrorss_and_mrms_io.FIELD_NAME_BY_PAIR_KEY]
    radar_height_by_pair_m_asl = file_dictionary[
        myrorss_and_mrms_io.HEIGHT_BY_PAIR_KEY]

    num_radar_fields = len(radar_field_name_by_pair)
    num_unique_storm_times = len(unique_storm_times_unix_sec)
    image_file_names = [''] * num_unique_storm_times

    for i in range(num_unique_storm_times):
        this_time_string = time_conversion.unix_sec_to_string(
            unique_storm_times_unix_sec[i], TIME_FORMAT_FOR_LOG_MESSAGES)
        this_spc_date_string = time_conversion.time_to_spc_date_string(
            unique_spc_dates_unix_sec[i])

        these_storm_flags = numpy.logical_and(
            storm_object_table[tracking_utils.TIME_COLUMN].values ==
            unique_storm_times_unix_sec[i],
            storm_object_table[tracking_utils.SPC_DATE_COLUMN].values ==
            unique_spc_dates_unix_sec[i])
        these_storm_indices = numpy.where(these_storm_flags)[0]

        this_num_storms = len(these_storm_indices)
        this_4d_image_matrix = numpy.full(
            (num_radar_fields, this_num_storms, num_rows_per_image,
             num_columns_per_image), numpy.nan)

        for j in range(num_radar_fields):
            if radar_file_names_2d_list[i][j] is None:
                continue

            print (
                'Extracting images for "{0:s}" at {1:d} metres ASL and '
                '{2:s}...').format(
                    radar_field_name_by_pair[j],
                    int(radar_height_by_pair_m_asl[j]), this_time_string)

            this_metadata_dict = (
                myrorss_and_mrms_io.read_metadata_from_raw_file(
                    radar_file_names_2d_list[i][j],
                    data_source=radar_data_source))

            this_sparse_grid_table = (
                myrorss_and_mrms_io.read_data_from_sparse_grid_file(
                    radar_file_names_2d_list[i][j],
                    field_name_orig=this_metadata_dict[
                        myrorss_and_mrms_io.FIELD_NAME_COLUMN_ORIG],
                    data_source=radar_data_source,
                    sentinel_values=this_metadata_dict[
                        radar_utils.SENTINEL_VALUE_COLUMN]))

            this_radar_matrix, _, _ = radar_s2f.sparse_to_full_grid(
                this_sparse_grid_table, this_metadata_dict)
            this_radar_matrix[numpy.isnan(this_radar_matrix)] = 0.

            these_storm_center_rows, these_storm_center_columns = (
                _center_points_latlng_to_rowcol(
                    center_latitudes_deg=storm_object_table[
                        tracking_utils.CENTROID_LAT_COLUMN].values[
                            these_storm_indices],
                    center_longitudes_deg=storm_object_table[
                        tracking_utils.CENTROID_LNG_COLUMN].values[
                            these_storm_indices],
                    nw_grid_point_lat_deg=
                    this_metadata_dict[radar_utils.NW_GRID_POINT_LAT_COLUMN],
                    nw_grid_point_lng_deg=
                    this_metadata_dict[radar_utils.NW_GRID_POINT_LNG_COLUMN],
                    lat_spacing_deg=
                    this_metadata_dict[radar_utils.LAT_SPACING_COLUMN],
                    lng_spacing_deg=
                    this_metadata_dict[radar_utils.LNG_SPACING_COLUMN]))

            for k in range(this_num_storms):
                this_4d_image_matrix[j, k, :, :] = extract_radar_subgrid(
                    this_radar_matrix,
                    center_row_index=these_storm_center_rows[k],
                    center_column_index=these_storm_center_columns[k],
                    num_rows_in_subgrid=num_rows_per_image,
                    num_columns_in_subgrid=num_columns_per_image)

        image_file_names[i] = 'storm_images_{0:s}_{1:s}.p'.format(
            this_spc_date_string, this_time_string)
        print 'Writing images to "{0:s}"...\n'.format(image_file_names[i])

        these_storm_ids = storm_object_table[
            tracking_utils.STORM_ID_COLUMN].values[these_storm_indices].tolist()
        write_storm_images(
            image_file_names[i], storm_ids=these_storm_ids,
            radar_field_name_by_pair=radar_field_name_by_pair,
            radar_height_by_pair_m_asl=radar_height_by_pair_m_asl,
            image_matrix=this_4d_image_matrix)

    return image_file_names


def write_storm_images(
        pickle_file_name, image_matrix, storm_ids, radar_field_name_by_pair,
        radar_height_by_pair_m_asl):
    """Writes storm-centered radar images (subgrids) to Pickle file.

    These images should be created by `get_images_for_storm_objects`.

    S = number of storm objects
    F = number of radar fields (variable/height pairs)
    M = number of rows in each image
    N = number of columns in each image

    :param pickle_file_name: Path to output file.
    :param image_matrix: See documentation for _check_storm_images.
    :param storm_ids: See doc for _check_storm_images.
    :param radar_field_name_by_pair: See doc for _check_storm_images.
    :param radar_height_by_pair_m_asl: See doc for _check_storm_images.
    """

    _check_storm_images(
        image_matrix=image_matrix, storm_ids=storm_ids,
        radar_field_name_by_pair=radar_field_name_by_pair,
        radar_height_by_pair_m_asl=radar_height_by_pair_m_asl)

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(image_matrix, pickle_file_handle)
    pickle.dump(storm_ids, pickle_file_handle)
    pickle.dump(radar_field_name_by_pair, pickle_file_handle)
    pickle.dump(radar_height_by_pair_m_asl, pickle_file_handle)
    pickle_file_handle.close()


def read_storm_images(pickle_file_name):
    """Reads storm-centered radar images (subgrids) from Pickle file.

    :param pickle_file_name: Path to output file.
    :return: image_matrix: See documentation for _check_storm_images.
    :return: storm_ids: See doc for _check_storm_images.
    :return: radar_field_name_by_pair: See doc for _check_storm_images.
    :return: radar_height_by_pair_m_asl: See doc for _check_storm_images.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    image_matrix = pickle.load(pickle_file_handle)
    storm_ids = pickle.load(pickle_file_handle)
    radar_field_name_by_pair = pickle.load(pickle_file_handle)
    radar_height_by_pair_m_asl = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    _check_storm_images(
        image_matrix=image_matrix, storm_ids=storm_ids,
        radar_field_name_by_pair=radar_field_name_by_pair,
        radar_height_by_pair_m_asl=radar_height_by_pair_m_asl)
