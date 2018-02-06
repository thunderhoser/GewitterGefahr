"""Methods for extracting radar subgrids.

These are usually centered around a storm object.
"""

import os.path
import pickle
import numpy
from gewittergefahr.gg_io import myrorss_and_mrms_io
from gewittergefahr.gg_io import gridrad_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import gridrad_utils
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TIME_FORMAT = '%Y-%m-%d-%H%M%S'

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

DEFAULT_FIELDS_FOR_MYRORSS_AND_MRMS = [
    radar_utils.ECHO_TOP_18DBZ_NAME, radar_utils.ECHO_TOP_50DBZ_NAME,
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME,
    radar_utils.REFL_COLUMN_MAX_NAME, radar_utils.MESH_NAME,
    radar_utils.REFL_0CELSIUS_NAME, radar_utils.REFL_M10CELSIUS_NAME,
    radar_utils.REFL_M20CELSIUS_NAME, radar_utils.REFL_LOWEST_ALTITUDE_NAME,
    radar_utils.SHI_NAME, radar_utils.VIL_NAME]

AZIMUTHAL_SHEAR_FIELD_NAMES = [
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME]

# TODO(thunderhoser): Deal with dual-pol variables in GridRad and the fact that
# they might be missing.
DEFAULT_FIELDS_FOR_GRIDRAD = [
    radar_utils.REFL_NAME, radar_utils.SPECTRUM_WIDTH_NAME,
    radar_utils.VORTICITY_NAME, radar_utils.DIVERGENCE_NAME]

DEFAULT_HEIGHTS_FOR_GRIDRAD_M_ASL = numpy.array(
    [1000, 2000, 3000, 4000, 5000, 8000, 10000, 12000], dtype=int)


def _center_points_latlng_to_rowcol(
        center_latitudes_deg, center_longitudes_deg, nw_grid_point_lat_deg,
        nw_grid_point_lng_deg, lat_spacing_deg, lng_spacing_deg):
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


def _get_rowcol_indices_for_subgrid(
        num_rows_in_full_grid, num_columns_in_full_grid, center_row_index,
        center_column_index, num_rows_in_subgrid, num_columns_in_subgrid):
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
        image_matrix, storm_ids, radar_field_name, radar_height_m_asl):
    """Checks storm-centered radar images for errors.

    These images should be created by `get_images_for_storm_objects`.

    K = number of storm objects
    M = number of rows in each image
    N = number of columns in each image

    :param image_matrix: K-by-M-by-N numpy array with image for each storm
        object.
    :param storm_ids: length-K list of storm IDs (strings).
    :param radar_field_name: Name of radar field.
    :param radar_height_m_asl: Height (metres above sea level) of radar field.
    """

    error_checking.assert_is_string_list(storm_ids)
    error_checking.assert_is_numpy_array(
        numpy.array(storm_ids), num_dimensions=1)
    num_storm_objects = len(storm_ids)

    radar_utils.check_field_name(radar_field_name)
    error_checking.assert_is_geq(radar_height_m_asl, 0)

    error_checking.assert_is_numpy_array(image_matrix, num_dimensions=3)
    error_checking.assert_is_numpy_array(
        image_matrix, exact_dimensions=numpy.array(
            [num_storm_objects, image_matrix.shape[1], image_matrix.shape[2]]))


def extract_radar_subgrid(
        field_matrix, center_row_index, center_column_index,
        num_rows_in_subgrid, num_columns_in_subgrid):
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


def extract_storm_images_myrorss_or_mrms(
        storm_object_table, radar_source, top_radar_dir_name,
        top_output_dir_name, num_rows_per_image=DEFAULT_NUM_ROWS_PER_IMAGE,
        num_columns_per_image=DEFAULT_NUM_COLUMNS_PER_IMAGE,
        radar_field_names=DEFAULT_FIELDS_FOR_MYRORSS_AND_MRMS,
        reflectivity_heights_m_asl=None):
    """Extracts radar image (subgrid) for each field, height, and storm object.

    In this case, radar data must be from either MYRORSS or MRMS.

    Images for storm object s will have the same center as s.

    N = number of storm objects
    F = number of radar fields
    P = number of field/height pairs
    T = number of time steps with storm objects

    :param storm_object_table: N-row pandas DataFrame with the following
        columns.  Each row is one storm object.
    storm_object_table.storm_id: String ID for storm cell.
    storm_object_table.unix_time_sec: Valid time.
    storm_object_table.spc_date_unix_sec: SPC date.
    storm_object_table.centroid_lat_deg: Latitude (deg N) of storm centroid.
    storm_object_table.centroid_lng_deg: Longitude (deg E) of storm centroid.

    :param radar_source: Source of radar data (either "myrorss" or "mrms").
    :param top_radar_dir_name: [input] Name of top-level directory with radar
        data from the given source.
    :param top_output_dir_name: [output] Name of top-level directory for storm-
        centered radar images.
    :param num_rows_per_image: Number of rows in each image.  We recommend that
        you make this a power of 2 (examples: 8, 16, 32, 64, etc.).
    :param num_columns_per_image: Number of columns in each image.  You should
        also make this a power of 2.
    :param radar_field_names: length-F list with names of radar fields.
    :param reflectivity_heights_m_asl: 1-D numpy array of heights (metres above
        sea level) for the field "reflectivity_dbz".  If "reflectivity_dbz" is
        not in `radar_field_names`, you can leave this as None.
    :return: image_file_names_2d_list: T-by-P list of paths to output files (one
        for each time step and field/height pair).
    """

    # Error-checking.
    error_checking.assert_is_integer(num_rows_per_image)
    error_checking.assert_is_geq(num_rows_per_image, MIN_ROWS_PER_IMAGE)
    error_checking.assert_is_integer(num_columns_per_image)
    error_checking.assert_is_geq(num_columns_per_image, MIN_COLUMNS_PER_IMAGE)

    # Find radar files.
    spc_date_strings = [
        time_conversion.time_to_spc_date_string(t)
        for t in storm_object_table[tracking_utils.SPC_DATE_COLUMN].values]

    file_dictionary = myrorss_and_mrms_io.find_many_raw_files(
        desired_times_unix_sec=
        storm_object_table[tracking_utils.TIME_COLUMN].values,
        spc_date_strings=spc_date_strings, data_source=radar_source,
        field_names=radar_field_names, top_directory_name=top_radar_dir_name,
        reflectivity_heights_m_asl=reflectivity_heights_m_asl)

    radar_file_names_2d_list = file_dictionary[
        myrorss_and_mrms_io.RADAR_FILE_NAME_LIST_KEY]
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

    # Initialize output.
    num_field_height_pairs = len(radar_field_name_by_pair)
    num_valid_times = len(valid_times_unix_sec)
    image_file_names_2d_list = (
        [[''] * num_field_height_pairs] * num_valid_times)

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
            if radar_file_names_2d_list[i][j] is None:
                continue

            this_3d_image_matrix = numpy.full(
                (this_num_storms, num_rows_per_image, num_columns_per_image),
                numpy.nan)

            print (
                'Extracting images for "{0:s}" at {1:d} metres ASL and '
                '{2:s}...').format(
                    radar_field_name_by_pair[j],
                    numpy.round(int(radar_height_by_pair_m_asl[j])),
                    valid_time_strings[i])

            # Read data for [j]th field/height pair at [i]th time step.
            this_metadata_dict = (
                myrorss_and_mrms_io.read_metadata_from_raw_file(
                    radar_file_names_2d_list[i][j], data_source=radar_source))

            this_sparse_grid_table = (
                myrorss_and_mrms_io.read_data_from_sparse_grid_file(
                    radar_file_names_2d_list[i][j],
                    field_name_orig=this_metadata_dict[
                        myrorss_and_mrms_io.FIELD_NAME_COLUMN_ORIG],
                    data_source=radar_source,
                    sentinel_values=this_metadata_dict[
                        radar_utils.SENTINEL_VALUE_COLUMN]))

            this_radar_matrix, _, _ = radar_s2f.sparse_to_full_grid(
                this_sparse_grid_table, this_metadata_dict)
            this_radar_matrix[numpy.isnan(this_radar_matrix)] = 0.

            # Extract storm-centered radar images for [j]th field/height pair at
            # [i]th time step.
            these_storm_center_rows, these_storm_center_columns = (
                _center_points_latlng_to_rowcol(
                    center_latitudes_deg=storm_object_table[
                        tracking_utils.CENTROID_LAT_COLUMN].values[
                            these_storm_indices],
                    center_longitudes_deg=storm_object_table[
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
                this_3d_image_matrix[k, :, :] = extract_radar_subgrid(
                    this_radar_matrix,
                    center_row_index=these_storm_center_rows[k],
                    center_column_index=these_storm_center_columns[k],
                    num_rows_in_subgrid=num_rows_per_image,
                    num_columns_in_subgrid=num_columns_per_image)

            # Write storm-centered radar images for [j]th field/height pair at
            # [i]th time step.
            image_file_names_2d_list[i][j] = find_storm_image_file(
                top_directory_name=top_output_dir_name,
                unix_time_sec=valid_times_unix_sec[i],
                spc_date_string=spc_date_strings[i],
                radar_field_name=radar_field_name_by_pair[j],
                radar_height_m_asl=radar_height_by_pair_m_asl[j],
                raise_error_if_missing=False)

            print 'Writing images to "{0:s}"...\n'.format(
                image_file_names_2d_list[i][j])
            write_storm_images(
                image_file_names_2d_list[i][j],
                image_matrix=this_3d_image_matrix, storm_ids=these_storm_ids,
                radar_field_name=radar_field_name_by_pair[j],
                radar_height_m_asl=radar_height_by_pair_m_asl[j])

    return image_file_names_2d_list


def extract_storm_images_gridrad(
        storm_object_table, top_radar_dir_name, top_output_dir_name,
        num_rows_per_image=DEFAULT_NUM_ROWS_PER_IMAGE,
        num_columns_per_image=DEFAULT_NUM_COLUMNS_PER_IMAGE,
        radar_field_names=DEFAULT_FIELDS_FOR_GRIDRAD,
        radar_heights_m_asl=DEFAULT_HEIGHTS_FOR_GRIDRAD_M_ASL):
    """Extracts radar image (subgrid) for each field, height, and storm object.

    In this case, radar data must be from GridRad.

    Images for storm object s will have the same center as s.

    N = number of storm objects
    F = number of radar fields
    H = number of radar heights
    T = number of time steps with storm objects

    :param storm_object_table: See documentation for
        `extract_storm_images_myrorss_or_mrms`.
    :param top_radar_dir_name: See doc for
        `extract_storm_images_myrorss_or_mrms`.
    :param top_output_dir_name: See doc for
        `extract_storm_images_myrorss_or_mrms`.
    :param num_rows_per_image: See doc for
        `extract_storm_images_myrorss_or_mrms`.
    :param num_columns_per_image: See doc for
        `extract_storm_images_myrorss_or_mrms`.
    :param radar_field_names: length-F list with names of radar fields.
    :param radar_heights_m_asl: length-H numpy array of radar heights (metres
        above sea level).
    :return: image_file_names_3d_list: T-by-F-by-H list of paths to output files
        (one for each time step, field, and height).
    """

    # Error-checking.
    error_checking.assert_is_integer(num_rows_per_image)
    error_checking.assert_is_geq(num_rows_per_image, MIN_ROWS_PER_IMAGE)
    error_checking.assert_is_integer(num_columns_per_image)
    error_checking.assert_is_geq(num_columns_per_image, MIN_COLUMNS_PER_IMAGE)

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

    # Initialize output.
    num_fields = len(radar_field_names)
    num_heights = len(radar_heights_m_asl)
    num_valid_times = len(valid_times_unix_sec)
    image_file_names_3d_list = (
        [[[''] * num_heights] * num_fields] * num_valid_times)

    for i in range(num_valid_times):

        # Read metadata for [i]th valid time and find storm objects at [i]th
        # valid time.
        this_metadata_dict = gridrad_io.read_metadata_from_full_grid_file(
            radar_file_names[i])

        these_storm_indices = numpy.where(
            storm_object_table[tracking_utils.TIME_COLUMN].values ==
            valid_times_unix_sec[i])[0]
        these_storm_ids = storm_object_table[
            tracking_utils.STORM_ID_COLUMN].values[these_storm_indices].tolist()
        this_num_storms = len(these_storm_indices)

        for j in range(num_fields):

            # Read data for [j]th field at [i]th valid time.
            print 'Reading "{0:s}" from file "{1:s}"...'.format(
                radar_field_names[j], valid_time_strings[i])

            radar_matrix_this_field, these_grid_point_heights_m_asl, _, _ = (
                gridrad_io.read_field_from_full_grid_file(
                    radar_file_names[i], field_name=radar_field_names[j],
                    metadata_dict=this_metadata_dict))

            these_grid_point_heights_m_asl = numpy.round(
                these_grid_point_heights_m_asl).astype(int)
            these_height_indices_to_keep = numpy.array(
                [these_grid_point_heights_m_asl.tolist().index(h)
                 for h in radar_heights_m_asl], dtype=int)
            del these_grid_point_heights_m_asl

            radar_matrix_this_field = (
                radar_matrix_this_field[these_height_indices_to_keep, :, :])
            radar_matrix_this_field[numpy.isnan(radar_matrix_this_field)] = 0.

            for k in range(num_heights):
                this_3d_image_matrix = numpy.full(
                    (this_num_storms, num_rows_per_image,
                     num_columns_per_image),
                    numpy.nan)

                print (
                    'Extracting images for "{0:s}" at {1:d} metres ASL and '
                    '{2:s}...').format(
                        radar_field_names[j],
                        numpy.round(int(radar_heights_m_asl[k])),
                        valid_time_strings[i])

                # Extract storm-centered radar images for [j]th field at [k]th
                # height and [i]th valid time.
                these_storm_center_rows, these_storm_center_columns = (
                    _center_points_latlng_to_rowcol(
                        center_latitudes_deg=storm_object_table[
                            tracking_utils.CENTROID_LAT_COLUMN].values[
                                these_storm_indices],
                        center_longitudes_deg=storm_object_table[
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
                    this_3d_image_matrix[m, :, :] = extract_radar_subgrid(
                        numpy.flipud(radar_matrix_this_field[k, :, :]),
                        center_row_index=these_storm_center_rows[m],
                        center_column_index=these_storm_center_columns[m],
                        num_rows_in_subgrid=num_rows_per_image,
                        num_columns_in_subgrid=num_columns_per_image)

                # Write storm-centered radar images for [j]th field at [k]th
                # height and [i]th valid time.
                image_file_names_3d_list[i][j][k] = find_storm_image_file(
                    top_directory_name=top_output_dir_name,
                    unix_time_sec=valid_times_unix_sec[i],
                    spc_date_string=time_conversion.time_to_spc_date_string(
                        valid_times_unix_sec[i]),
                    radar_field_name=radar_field_names[j],
                    radar_height_m_asl=radar_heights_m_asl[k],
                    raise_error_if_missing=False)

                print 'Writing images to "{0:s}"...\n'.format(
                    image_file_names_3d_list[i][j][k])
                write_storm_images(
                    image_file_names_3d_list[i][j][k],
                    image_matrix=this_3d_image_matrix,
                    storm_ids=these_storm_ids,
                    radar_field_name=radar_field_names[j],
                    radar_height_m_asl=radar_heights_m_asl[k])

    return image_file_names_3d_list


def find_storm_image_file(
        top_directory_name, unix_time_sec, spc_date_string, radar_field_name,
        radar_height_m_asl, raise_error_if_missing=True):
    """Finds file with storm-centered radar images.

    :param top_directory_name: Name of top-level directory with files containing
        storm-centered radar images.
    :param unix_time_sec: Valid time.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param radar_field_name: Name of radar field.
    :param radar_height_m_asl: Height (metres above sea level) of radar field.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        will raise error.
    :return: storm_image_file_name: Path to image file.  If file is missing and
        raise_error_if_missing = False, this is the *expected* path.
    """

    error_checking.assert_is_string(top_directory_name)
    _ = time_conversion.spc_date_string_to_unix_sec(spc_date_string)
    error_checking.assert_is_string(radar_field_name)
    error_checking.assert_is_greater(radar_height_m_asl, 0)
    error_checking.assert_is_boolean(raise_error_if_missing)

    storm_image_file_name = (
        '{0:s}/{1:s}/{2:s}/{3:s}/{4:05d}_metres_asl/'
        'storm_images_{5:s}.p').format(
            top_directory_name, spc_date_string[:4], spc_date_string,
            radar_field_name, numpy.round(int(radar_height_m_asl)),
            time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT))

    if raise_error_if_missing and not os.path.isfile(storm_image_file_name):
        error_string = (
            'Cannot find file with storm-centered images.  Expected at: '
            '{0:s}').format(storm_image_file_name)
        raise ValueError(error_string)

    return storm_image_file_name


def write_storm_images(
        pickle_file_name, image_matrix, storm_ids, radar_field_name,
        radar_height_m_asl):
    """Writes storm-centered radar images (subgrids) to Pickle file.

    These images should be created by `get_images_for_storm_objects`.

    :param pickle_file_name: Path to output file.
    :param image_matrix: See documentation for _check_storm_images.
    :param storm_ids: See doc for _check_storm_images.
    :param radar_field_name: See doc for _check_storm_images.
    :param radar_height_m_asl: See doc for _check_storm_images.
    """

    _check_storm_images(
        image_matrix=image_matrix, storm_ids=storm_ids,
        radar_field_name=radar_field_name,
        radar_height_m_asl=radar_height_m_asl)

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(image_matrix, pickle_file_handle)
    pickle.dump(storm_ids, pickle_file_handle)
    pickle.dump(radar_field_name, pickle_file_handle)
    pickle.dump(radar_height_m_asl, pickle_file_handle)
    pickle_file_handle.close()


def read_storm_images(pickle_file_name):
    """Reads storm-centered radar images (subgrids) from Pickle file.

    :param pickle_file_name: Path to output file.
    :return: image_matrix: See documentation for _check_storm_images.
    :return: storm_ids: See doc for _check_storm_images.
    :return: radar_field_name: See doc for _check_storm_images.
    :return: radar_height_m_asl: See doc for _check_storm_images.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    image_matrix = pickle.load(pickle_file_handle)
    storm_ids = pickle.load(pickle_file_handle)
    radar_field_name = pickle.load(pickle_file_handle)
    radar_height_m_asl = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    _check_storm_images(
        image_matrix=image_matrix, storm_ids=storm_ids,
        radar_field_name=radar_field_name,
        radar_height_m_asl=radar_height_m_asl)

    return image_matrix, storm_ids, radar_field_name, radar_height_m_asl
